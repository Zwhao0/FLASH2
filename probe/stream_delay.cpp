#include <nccl.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>

namespace {

template <typename Function>
Function resolve_next(const char* name) noexcept {
    void* symbol = dlsym(RTLD_NEXT, name);
    if (symbol == nullptr) {
        // PyTorch loads NCCL into a local link-map; RTLD_NEXT alone can miss it.
        static void* handle = []() noexcept -> void* {
            void* h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
            return h ? h : dlopen("libnccl.so", RTLD_NOW | RTLD_LOCAL);
        }();
        if (handle) symbol = dlsym(handle, name);
    }
    return reinterpret_cast<Function>(symbol);
}

struct ThreadGroupState {
    unsigned int depth = 0;
    bool p2p_attempted = false;
};

thread_local ThreadGroupState group_state;

int parse_integer_env(const char* name, int fallback) noexcept {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return fallback;
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(value);
}

bool selected_rank() noexcept {
    const int rank = parse_integer_env("RANK", -1);
    const int stall_rank = parse_integer_env("FLASH_STALL_RANK", 0);
    return rank >= 0 && rank == stall_rank;
}

bool selected_operation(bool allreduce, ncclComm_t comm) noexcept {
    const char* kind = std::getenv("FLASH_STALL_KIND");
    const bool wants_allreduce = kind != nullptr && std::string(kind) == "allreduce";
    if (wants_allreduce != allreduce) return false;
    const int desired_size = parse_integer_env("FLASH_STALL_COMM_SIZE", 0);
    if (desired_size > 0) {
        using Function = decltype(&ncclCommCount);
        static Function count = resolve_next<Function>("ncclCommCount");
        int size = 0;
        if (count == nullptr || count(comm, &size) != ncclSuccess || size != desired_size) return false;
    }
    return true;
}

double read_delay_ms() noexcept {
    const char* file_path = std::getenv("FLASH_STALL_FILE");
    if (file_path == nullptr || *file_path == '\0') return 0.0;
    try {
        std::ifstream input(file_path);
        if (!input) return 0.0;
        std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(text.c_str(), &end);
        if (errno != 0 || end == text.c_str() || !std::isfinite(value) || value <= 0.0) {
            return 0.0;
        }
        while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
        if (*end != '\0') return 0.0;

        // Keep the later sleep_for conversion within steady_clock's representable range.
        using ClockDuration = std::chrono::steady_clock::duration;
        const long double max_seconds =
            static_cast<long double>(ClockDuration::max().count()) *
            static_cast<long double>(ClockDuration::period::num) /
            static_cast<long double>(ClockDuration::period::den);
        const long double max_ms = max_seconds * 1000.0L;
        if (static_cast<long double>(value) > max_ms) return 0.0;
        return value;
    } catch (...) {
        return 0.0;
    }
}

void CUDART_CB stream_stall_callback(void* user_data) noexcept {
    std::unique_ptr<double> delay_ms(static_cast<double*>(user_data));
    if (!delay_ms) return;
    try {
        std::this_thread::sleep_for(
            std::chrono::duration<double, std::milli>(*delay_ms));
    } catch (...) {
    }
}

void queue_stall(cudaStream_t stream) noexcept {
    if (!selected_rank()) return;
    const double delay_ms = read_delay_ms();
    if (!(delay_ms > 0.0)) return;

    double* callback_arg = new (std::nothrow) double(delay_ms);
    if (callback_arg == nullptr) return;
    if (cudaLaunchHostFunc(stream, stream_stall_callback, callback_arg) != cudaSuccess) {
        delete callback_arg;
    }
}

void before_p2p(cudaStream_t stream) noexcept {
    if (group_state.depth == 0) {
        // Each standalone Send/Recv gets one independent chance to inject a delay.
        queue_stall(stream);
        return;
    }
    if (group_state.p2p_attempted) return;

    // Consume the group's single attempt even if the file is absent or CUDA rejects it.
    group_state.p2p_attempted = true;
    queue_stall(stream);
}

}  // namespace

extern "C" ncclResult_t ncclGroupStart(void) {
    using Function = decltype(&ncclGroupStart);
    static Function real = resolve_next<Function>("ncclGroupStart");
    if (real == nullptr) return ncclSystemError;

    const ncclResult_t result = real();
    if (result == ncclSuccess) {
        if (group_state.depth == 0) group_state.p2p_attempted = false;
        ++group_state.depth;
    }
    return result;
}

extern "C" ncclResult_t ncclGroupEnd(void) {
    using Function = decltype(&ncclGroupEnd);
    static Function real = resolve_next<Function>("ncclGroupEnd");
    if (real == nullptr) return ncclSystemError;

    const ncclResult_t result = real();
    if (group_state.depth > 0) {
        --group_state.depth;
        if (group_state.depth == 0) group_state.p2p_attempted = false;
    }
    return result;
}

extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count,
                                  ncclDataType_t datatype, int peer, ncclComm_t comm,
                                  cudaStream_t stream) {
    using Function = decltype(&ncclSend);
    static Function real = resolve_next<Function>("ncclSend");
    if (real == nullptr) return ncclSystemError;

    if (selected_operation(false, comm)) before_p2p(stream);
    return real(sendbuff, count, datatype, peer, comm, stream);
}

extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count,
                                  ncclDataType_t datatype, int peer, ncclComm_t comm,
                                  cudaStream_t stream) {
    using Function = decltype(&ncclRecv);
    static Function real = resolve_next<Function>("ncclRecv");
    if (real == nullptr) return ncclSystemError;

    if (selected_operation(false, comm)) before_p2p(stream);
    return real(recvbuff, count, datatype, peer, comm, stream);
}

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                                      size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm_t comm,
                                      cudaStream_t stream) {
    using Function = decltype(&ncclAllReduce);
    static Function real = resolve_next<Function>("ncclAllReduce");
    if (real == nullptr) return ncclSystemError;
    if (selected_operation(true, comm)) before_p2p(stream);
    return real(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

