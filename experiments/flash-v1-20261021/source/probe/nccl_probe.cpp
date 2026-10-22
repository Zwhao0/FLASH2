#define _GNU_SOURCE

#include <nccl.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(NCCL_VERSION_CODE) && (NCCL_VERSION_CODE >= 21400)
#define FLASH_NCCL_HAS_INIT_RANK_CONFIG 1
#elif defined(NCCL_MAJOR) && defined(NCCL_MINOR) && \
      ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 14)))
#define FLASH_NCCL_HAS_INIT_RANK_CONFIG 1
#else
#define FLASH_NCCL_HAS_INIT_RANK_CONFIG 0
#endif

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kRecordCapacity = 8192;
constexpr char kMagic[8] = {'F', 'L', 'A', 'S', 'H', 'N', 'C', 'L'};
constexpr uint32_t kHeaderOverflowed = 1u << 0;
constexpr uint32_t kRecordGrouped = 1u << 0;
constexpr uint32_t kRecordTimingRequested = 1u << 1;
constexpr uint32_t kRecordTimingValid = 1u << 2;
constexpr uint32_t kRecordTimingError = 1u << 3;

enum OperationKind : int32_t {
    kAllReduce = 1,
    kAllGather = 2,
    kReduceScatter = 3,
    kBroadcast = 4,
    kReduce = 5,
    kSend = 6,
    kRecv = 7,
};

struct alignas(8) SharedHeader {
    char magic[8];                // 0
    uint32_t version;             // 8
    uint32_t header_size;         // 12
    uint32_t record_size;         // 16
    uint32_t capacity;            // 20
    uint32_t mode;                // 24; atomic control word: 0 or 1
    int32_t rank;                 // 28
    uint64_t write_seq;           // 32; number of committed records
    uint64_t dropped;             // 40; overwritten record count
    uint64_t heartbeat_ns;        // 48; CLOCK_MONOTONIC timestamp
    uint32_t pid;                 // 56
    uint32_t flags;               // 60
};

struct alignas(8) SharedRecord {
    uint64_t seq;                 // 0; commit marker, 1-based
    uint64_t start_ns;            // 8; CLOCK_MONOTONIC host timestamp
    uint64_t end_ns;              // 16; CLOCK_MONOTONIC host timestamp
    uint64_t cuda_duration_ns;    // 24; CUDA event elapsed time, or 0
    uint64_t comm_hash;           // 32; FNV-1a over ncclUniqueId
    uint64_t count;               // 40; NCCL count argument
    uint64_t group_id;            // 48; 0 for ungrouped calls
    int32_t datatype;             // 56; raw ncclDataType_t value
    int32_t redop;                // 60; raw ncclRedOp_t value, -1 if absent
    int32_t op;                   // 64; OperationKind above
    int32_t peer_root;            // 68; peer/root, -1 if absent
    int32_t rank;                 // 72; global rank from environment
    int32_t comm_size;            // 76; nranks from communicator init, -1 if unknown
    int32_t local_rank;           // 80; rank argument from communicator init, -1 if unknown
    int32_t result;               // 84; op result, or GroupEnd result for groups
    uint32_t op_index;            // 88; zero-based within group
    uint32_t op_count;            // 92; number of operations in group
    uint32_t stream_index;        // 96; zero-based stream index within group
    uint32_t stream_count;        // 100; distinct streams in group
    uint32_t flags;               // 104; see ABI.md
    int32_t group_result;         // 108; actual outer GroupEnd result; 0 if ungrouped
    uint8_t reserved[16];         // 112
};

static_assert(sizeof(SharedHeader) == 64, "shared header ABI changed");
static_assert(offsetof(SharedHeader, mode) == 24, "shared header mode offset changed");
static_assert(offsetof(SharedHeader, heartbeat_ns) == 48, "heartbeat offset changed");
static_assert(sizeof(SharedRecord) == 128, "shared record ABI changed");
static_assert(offsetof(SharedRecord, cuda_duration_ns) == 24, "duration offset changed");
static_assert(offsetof(SharedRecord, comm_size) == 76, "record comm_size offset changed");
static_assert(offsetof(SharedRecord, flags) == 104, "record flags offset changed");

uint64_t monotonic_ns() noexcept {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

int parse_env_int(const char* name, int fallback) noexcept {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed < INT32_MIN || parsed > INT32_MAX) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

int global_rank_from_env() noexcept {
    const char* names[] = {"RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID",
                           "PMI_RANK", "LOCAL_RANK"};
    for (const char* name : names) {
        const int rank = parse_env_int(name, INT32_MIN);
        if (rank != INT32_MIN) return rank;
    }
    return 0;
}

bool valid_run_id(const char* run_id) noexcept {
    if (run_id == nullptr || *run_id == '\0') return false;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(run_id);
         *p != '\0'; ++p) {
        if (!( (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
               (*p >= '0' && *p <= '9') || *p == '_' || *p == '-' || *p == '.')) {
            return false;
        }
    }
    return true;
}

struct CommunicatorInfo {
    uint64_t hash = 0;
    int32_t comm_size = -1;
    int32_t local_rank = -1;
};

class ProbeState {
public:
    ProbeState() = default;
    ProbeState(const ProbeState&) = delete;
    ProbeState& operator=(const ProbeState&) = delete;

    ~ProbeState() {
        if (mapping_ != MAP_FAILED) munmap(mapping_, mapping_size_);
        if (fd_ >= 0) close(fd_);
        // The /dev/shm file deliberately remains for the controller to read.
    }

    void api_entry() noexcept {
        try {
            std::call_once(initialize_once_, [this]() { initialize(); });
        } catch (...) {
            return;
        }
        if (header_ != nullptr) {
            __atomic_store_n(&header_->heartbeat_ns, monotonic_ns(), __ATOMIC_RELEASE);
        }
    }

    bool enabled() const noexcept { return header_ != nullptr; }

    uint32_t mode() const noexcept {
        if (header_ == nullptr) return 0;
        const uint32_t value = __atomic_load_n(&header_->mode, __ATOMIC_ACQUIRE);
        return value == 1 ? 1u : 0u;
    }

    int rank() const noexcept { return rank_; }

    CommunicatorInfo comm_info(ncclComm_t comm) noexcept {
        try {
            std::lock_guard<std::mutex> lock(comm_mutex_);
            auto it = comm_hashes_.find(comm);
            return it == comm_hashes_.end() ? CommunicatorInfo{} : it->second;
        } catch (...) {
            return CommunicatorInfo{};
        }
    }

    void remember_comm(ncclComm_t comm, uint64_t hash, int comm_size,
                       int local_rank) noexcept {
        if (comm == nullptr) return;
        try {
            std::lock_guard<std::mutex> lock(comm_mutex_);
            comm_hashes_[comm] = CommunicatorInfo{
                hash, static_cast<int32_t>(comm_size), static_cast<int32_t>(local_rank)};
        } catch (...) {
        }
    }

    void publish(const SharedRecord& record) noexcept {
        if (header_ == nullptr || records_ == nullptr) return;
        try {
            std::lock_guard<std::mutex> lock(write_mutex_);
            const uint64_t previous = __atomic_load_n(&header_->write_seq, __ATOMIC_RELAXED);
            const uint64_t seq = previous + 1;
            SharedRecord* slot = &records_[(seq - 1) % header_->capacity];
            __atomic_store_n(&slot->seq, 0, __ATOMIC_RELAXED);
            std::memcpy(reinterpret_cast<char*>(slot) + sizeof(slot->seq),
                        reinterpret_cast<const char*>(&record) + sizeof(record.seq),
                        sizeof(SharedRecord) - sizeof(record.seq));
            __atomic_store_n(&slot->seq, seq, __ATOMIC_RELEASE);
            __atomic_store_n(&header_->write_seq, seq, __ATOMIC_RELEASE);
            const uint64_t overwritten = seq > header_->capacity ? seq - header_->capacity : 0;
            __atomic_store_n(&header_->dropped, overwritten, __ATOMIC_RELEASE);
            if (overwritten != 0) {
                __atomic_fetch_or(&header_->flags, kHeaderOverflowed, __ATOMIC_RELEASE);
            }
        } catch (...) {
        }
    }

private:
    void initialize() {
        const char* run_id = std::getenv("FLASH_RUN_ID");
        if (!valid_run_id(run_id)) return;

        rank_ = global_rank_from_env();
        const std::string path = std::string("/dev/shm/flash_") + run_id + "_" +
                                 std::to_string(rank_);
        fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd_ < 0) return;
        if (flock(fd_, LOCK_EX) != 0) return;

        mapping_size_ = sizeof(SharedHeader) +
                        static_cast<size_t>(kRecordCapacity) * sizeof(SharedRecord);
        struct stat st{};
        bool ok = fstat(fd_, &st) == 0;
        if (ok && static_cast<size_t>(st.st_size) != mapping_size_) {
            ok = ftruncate(fd_, static_cast<off_t>(mapping_size_)) == 0;
        }
        if (ok) {
            mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            ok = mapping_ != MAP_FAILED;
        }
        if (ok) {
            auto* candidate = static_cast<SharedHeader*>(mapping_);
            const bool valid = std::memcmp(candidate->magic, kMagic, sizeof(kMagic)) == 0 &&
                               candidate->version == kAbiVersion &&
                               candidate->header_size == sizeof(SharedHeader) &&
                               candidate->record_size == sizeof(SharedRecord) &&
                               candidate->capacity == kRecordCapacity;
            if (!valid) {
                std::memset(mapping_, 0, mapping_size_);
                std::memcpy(candidate->magic, kMagic, sizeof(kMagic));
                candidate->version = kAbiVersion;
                candidate->header_size = sizeof(SharedHeader);
                candidate->record_size = sizeof(SharedRecord);
                candidate->capacity = kRecordCapacity;
                const int configured_mode = parse_env_int("FLASH_PROBE_MODE", 0);
                candidate->mode = configured_mode == 1 ? 1u : 0u;
                candidate->rank = rank_;
                candidate->pid = static_cast<uint32_t>(getpid());
            } else {
                // Preserve a mode value written by the controller before first attach.
                if (__atomic_load_n(&candidate->mode, __ATOMIC_RELAXED) > 1) {
                    __atomic_store_n(&candidate->mode, 0u, __ATOMIC_RELAXED);
                }
                candidate->rank = rank_;
                candidate->pid = static_cast<uint32_t>(getpid());
            }
            header_ = candidate;
            records_ = reinterpret_cast<SharedRecord*>(
                static_cast<char*>(mapping_) + sizeof(SharedHeader));
        }
        flock(fd_, LOCK_UN);
        if (!ok) {
            if (mapping_ != MAP_FAILED) {
                munmap(mapping_, mapping_size_);
                mapping_ = MAP_FAILED;
            }
            close(fd_);
            fd_ = -1;
        }
    }

    std::once_flag initialize_once_;
    int fd_ = -1;
    int rank_ = 0;
    void* mapping_ = MAP_FAILED;
    size_t mapping_size_ = 0;
    SharedHeader* header_ = nullptr;
    SharedRecord* records_ = nullptr;
    std::mutex write_mutex_;
    std::mutex comm_mutex_;
    std::unordered_map<ncclComm_t, CommunicatorInfo> comm_hashes_;
};

ProbeState& probe() {
    static ProbeState state;
    return state;
}

void* nccl_library_handle() noexcept {
    static void* handle = []() noexcept -> void* {
        void* result = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
        if (result == nullptr) result = dlopen("libnccl.so", RTLD_NOW | RTLD_LOCAL);
        return result;
    }();
    return handle;
}

template <typename Function>
Function resolve_nccl(const char* name) noexcept {
    void* symbol = dlsym(RTLD_NEXT, name);
    if (symbol == nullptr) {
        void* handle = nccl_library_handle();
        if (handle != nullptr) symbol = dlsym(handle, name);
    }
    return reinterpret_cast<Function>(symbol);
}

uint64_t unique_id_hash(const ncclUniqueId& id) noexcept {
    constexpr uint64_t offset_basis = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;
    uint64_t hash = offset_basis;
    const auto* bytes = reinterpret_cast<const unsigned char*>(id.internal);
    for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        hash ^= bytes[i];
        hash *= prime;
    }
    return hash;
}

struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool start_created = false;
    bool stop_created = false;
    bool start_recorded = false;
    bool stop_recorded = false;
    bool error = false;
    uint64_t duration_ns = 0;
};

void destroy_events(EventPair& pair) noexcept {
    if (pair.stop_created) cudaEventDestroy(pair.stop);
    if (pair.start_created) cudaEventDestroy(pair.start);
    pair = EventPair{};
}

bool start_events(EventPair& pair, cudaStream_t stream) noexcept {
    if (cudaEventCreateWithFlags(&pair.start, cudaEventDefault) != cudaSuccess) {
        pair.error = true;
        return false;
    }
    pair.start_created = true;
    if (cudaEventCreateWithFlags(&pair.stop, cudaEventDefault) != cudaSuccess) {
        pair.error = true;
        return false;
    }
    pair.stop_created = true;
    if (cudaEventRecord(pair.start, stream) != cudaSuccess) {
        pair.error = true;
        return false;
    }
    pair.start_recorded = true;
    return true;
}

void record_stop(EventPair& pair, cudaStream_t stream) noexcept {
    if (!pair.start_recorded || !pair.stop_created) {
        pair.error = true;
        return;
    }
    if (cudaEventRecord(pair.stop, stream) == cudaSuccess) {
        pair.stop_recorded = true;
    } else {
        pair.error = true;
    }
}

bool synchronize_and_measure(EventPair& pair) noexcept {
    if (!pair.stop_recorded || cudaEventSynchronize(pair.stop) != cudaSuccess) {
        pair.error = true;
        return false;
    }
    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, pair.start, pair.stop) != cudaSuccess ||
        elapsed_ms < 0.0f) {
        pair.error = true;
        return false;
    }
    pair.duration_ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f + 0.5f);
    return true;
}

struct Operation {
    uint64_t comm_hash = 0;
    uint64_t count = 0;
    int32_t datatype = -1;
    int32_t redop = -1;
    int32_t kind = 0;
    int32_t peer_root = -1;
    int32_t comm_size = -1;
    int32_t local_rank = -1;
    int32_t result = static_cast<int32_t>(ncclSuccess);
    uint32_t stream_index = 0;
};

struct StreamTiming {
    cudaStream_t stream = nullptr;
    EventPair events;
};

struct GroupContext {
    bool recording = false;
    bool initialized = false;
    bool timing_requested = false;
    bool timing_error = false;
    uint64_t group_id = 0;
    uint64_t start_ns = 0;
    std::vector<Operation> operations;
    std::vector<StreamTiming> streams;

    void reset() noexcept {
        for (auto& stream : streams) destroy_events(stream.events);
        operations.clear();
        streams.clear();
        recording = false;
        initialized = false;
        timing_requested = false;
        timing_error = false;
        group_id = 0;
        start_ns = 0;
    }
};

struct ThreadContext {
    int group_depth = 0;
    GroupContext group;
};

thread_local ThreadContext thread_context;
std::atomic<uint64_t> next_group_id{0};

struct OperationTicket {
    bool valid = false;
    bool grouped = false;
    uint32_t op_index = 0;
    uint64_t start_ns = 0;
    uint64_t comm_hash = 0;
    uint64_t count = 0;
    int32_t datatype = -1;
    int32_t redop = -1;
    int32_t kind = 0;
    int32_t peer_root = -1;
    int32_t comm_size = -1;
    int32_t local_rank = -1;
    EventPair standalone_events;
    bool timing_requested = false;
};

uint32_t find_or_add_stream(GroupContext& group, cudaStream_t stream) {
    for (size_t i = 0; i < group.streams.size(); ++i) {
        if (group.streams[i].stream == stream) return static_cast<uint32_t>(i);
    }
    StreamTiming timing;
    timing.stream = stream;
    if (group.timing_requested && !start_events(timing.events, stream)) {
        group.timing_error = true;
    }
    group.streams.push_back(timing);
    return static_cast<uint32_t>(group.streams.size() - 1);
}

OperationTicket begin_operation(int32_t kind, uint64_t count, ncclDataType_t datatype,
                                int32_t redop, int32_t peer_root, ncclComm_t comm,
                                cudaStream_t stream) noexcept {
    OperationTicket ticket;
    try {
        ProbeState& state = probe();
        ticket.valid = state.enabled();
        ticket.grouped = thread_context.group_depth > 0;
        ticket.start_ns = monotonic_ns();
        if (!ticket.valid) return ticket;

        const CommunicatorInfo comm_info = state.comm_info(comm);
        ticket.comm_hash = comm_info.hash;
        ticket.comm_size = comm_info.comm_size;
        ticket.local_rank = comm_info.local_rank;
        ticket.count = count;
        ticket.datatype = static_cast<int32_t>(datatype);
        ticket.redop = redop;
        ticket.kind = kind;
        ticket.peer_root = peer_root;

        if (ticket.grouped) {
            GroupContext& group = thread_context.group;
            if (!group.initialized) {
                group.reset();
                group.recording = true;
                group.initialized = true;
            }
            if (group.operations.empty()) {
                group.start_ns = ticket.start_ns;
                group.group_id = next_group_id.fetch_add(1, std::memory_order_relaxed) + 1;
                group.timing_requested = state.mode() == 1;
            }
            ticket.timing_requested = group.timing_requested;
            const uint32_t stream_index = find_or_add_stream(group, stream);
            Operation operation;
            operation.comm_hash = ticket.comm_hash;
            operation.count = count;
            operation.datatype = ticket.datatype;
            operation.redop = redop;
            operation.kind = kind;
            operation.peer_root = peer_root;
            operation.comm_size = ticket.comm_size;
            operation.local_rank = ticket.local_rank;
            operation.stream_index = stream_index;
            ticket.op_index = static_cast<uint32_t>(group.operations.size());
            group.operations.push_back(operation);
        } else {
            ticket.timing_requested = state.mode() == 1;
            if (ticket.timing_requested && !start_events(ticket.standalone_events, stream)) {
                // Metadata is still recorded, with a timing-error flag.
            }
        }
    } catch (...) {
        ticket.valid = false;
    }
    return ticket;
}

void finish_operation(OperationTicket& ticket, ncclResult_t result) noexcept {
    if (!ticket.valid) return;
    try {
        if (!ticket.grouped) return;
        GroupContext& group = thread_context.group;
        if (ticket.op_index < group.operations.size()) {
            group.operations[ticket.op_index].result = static_cast<int32_t>(result);
        }
    } catch (...) {
    }
}

// Complete a standalone operation with its actual stream. This is separate so a failed
// NCCL call does not synchronize a stream that has no successful work to measure.
void finish_standalone(OperationTicket& ticket, ncclResult_t result, cudaStream_t stream) noexcept {
    if (!ticket.valid || ticket.grouped) {
        finish_operation(ticket, result);
        return;
    }
    try {
        uint64_t duration_ns = 0;
        uint64_t flags = 0;
        if (ticket.timing_requested) {
            flags |= kRecordTimingRequested;
            if (ticket.standalone_events.error) flags |= kRecordTimingError;
            if (result == ncclSuccess && ticket.standalone_events.start_recorded) {
                record_stop(ticket.standalone_events, stream);
                if (synchronize_and_measure(ticket.standalone_events)) {
                    duration_ns = ticket.standalone_events.duration_ns;
                    flags |= kRecordTimingValid;
                } else {
                    flags |= kRecordTimingError;
                }
            }
            if (ticket.standalone_events.error) flags |= kRecordTimingError;
        }
        const uint64_t end_ns = monotonic_ns();
        SharedRecord record{};
        record.start_ns = ticket.start_ns;
        record.end_ns = end_ns;
        record.cuda_duration_ns = duration_ns;
        record.comm_hash = ticket.comm_hash;
        record.count = ticket.count;
        record.datatype = ticket.datatype;
        record.redop = ticket.redop;
        record.op = ticket.kind;
        record.peer_root = ticket.peer_root;
        record.rank = probe().rank();
        record.comm_size = ticket.comm_size;
        record.local_rank = ticket.local_rank;
        record.result = static_cast<int32_t>(result);
        record.op_index = 0;
        record.op_count = 1;
        record.stream_index = 0;
        record.stream_count = 1;
        record.flags = static_cast<uint32_t>(flags);
        probe().publish(record);
        destroy_events(ticket.standalone_events);
    } catch (...) {
        destroy_events(ticket.standalone_events);
    }
}

void safe_api_entry() noexcept {
    try {
        probe().api_entry();
    } catch (...) {
    }
}

void finish_group(ncclResult_t group_result) noexcept {
    GroupContext& group = thread_context.group;
    if (!group.recording || group.operations.empty()) {
        group.reset();
        return;
    }

    try {
    uint32_t flags = kRecordGrouped;
    if (group.timing_requested) {
        flags |= kRecordTimingRequested;
        if (group_result == ncclSuccess) {
            // Queue all stop markers after GroupEnd has really returned. In particular,
            // no Send/Recv stream is synchronized while the NCCL group is still open.
            for (auto& stream : group.streams) {
                record_stop(stream.events, stream.stream);
            }
            bool all_valid = !group.streams.empty();
            uint64_t group_duration_ns = 0;
            for (auto& stream : group.streams) {
                const bool valid = synchronize_and_measure(stream.events);
                all_valid = all_valid && valid;
                if (valid) group_duration_ns = std::max(group_duration_ns,
                                                        stream.events.duration_ns);
                if (stream.events.error) group.timing_error = true;
            }
            if (all_valid) flags |= kRecordTimingValid;
            if (group.timing_error || !all_valid) flags |= kRecordTimingError;
            const uint64_t end_ns = monotonic_ns();
            for (size_t i = 0; i < group.operations.size(); ++i) {
                const Operation& operation = group.operations[i];
                SharedRecord record{};
                record.start_ns = group.start_ns;
                record.end_ns = end_ns;
                record.cuda_duration_ns = group_duration_ns;
                record.comm_hash = operation.comm_hash;
                record.count = operation.count;
                record.group_id = group.group_id;
                record.datatype = operation.datatype;
                record.redop = operation.redop;
                record.op = operation.kind;
                record.peer_root = operation.peer_root;
                record.rank = probe().rank();
                record.comm_size = operation.comm_size;
                record.local_rank = operation.local_rank;
                record.result = operation.result;
                record.op_index = static_cast<uint32_t>(i);
                record.op_count = static_cast<uint32_t>(group.operations.size());
                record.stream_index = operation.stream_index;
                record.stream_count = static_cast<uint32_t>(group.streams.size());
                record.flags = flags;
                record.group_result = static_cast<int32_t>(group_result);
                probe().publish(record);
            }
        } else {
            flags |= kRecordTimingError;
            const uint64_t end_ns = monotonic_ns();
            for (size_t i = 0; i < group.operations.size(); ++i) {
                const Operation& operation = group.operations[i];
                SharedRecord record{};
                record.start_ns = group.start_ns;
                record.end_ns = end_ns;
                record.comm_hash = operation.comm_hash;
                record.count = operation.count;
                record.group_id = group.group_id;
                record.datatype = operation.datatype;
                record.redop = operation.redop;
                record.op = operation.kind;
                record.peer_root = operation.peer_root;
                record.rank = probe().rank();
                record.comm_size = operation.comm_size;
                record.local_rank = operation.local_rank;
                record.result = operation.result;
                record.op_index = static_cast<uint32_t>(i);
                record.op_count = static_cast<uint32_t>(group.operations.size());
                record.stream_index = operation.stream_index;
                record.stream_count = static_cast<uint32_t>(group.streams.size());
                record.flags = flags;
                record.group_result = static_cast<int32_t>(group_result);
                probe().publish(record);
            }
        }
    } else {
        // Mode 0 records metadata and host submission boundaries without CUDA events.
        const uint64_t end_ns = monotonic_ns();
        for (size_t i = 0; i < group.operations.size(); ++i) {
            const Operation& operation = group.operations[i];
            SharedRecord record{};
            record.start_ns = group.start_ns;
            record.end_ns = end_ns;
            record.comm_hash = operation.comm_hash;
            record.count = operation.count;
            record.group_id = group.group_id;
            record.datatype = operation.datatype;
            record.redop = operation.redop;
            record.op = operation.kind;
            record.peer_root = operation.peer_root;
            record.rank = probe().rank();
            record.comm_size = operation.comm_size;
            record.local_rank = operation.local_rank;
            record.result = operation.result;
            record.op_index = static_cast<uint32_t>(i);
            record.op_count = static_cast<uint32_t>(group.operations.size());
            record.stream_index = operation.stream_index;
            record.stream_count = static_cast<uint32_t>(group.streams.size());
            record.flags = kRecordGrouped;
            record.group_result = static_cast<int32_t>(group_result);
            probe().publish(record);
        }
    }
    } catch (...) {
    }
    group.reset();
}

}  // namespace

extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                                          ncclUniqueId comm_id, int rank) {
    safe_api_entry();
    using Function = decltype(&ncclCommInitRank);
    static Function real = resolve_nccl<Function>("ncclCommInitRank");
    if (real == nullptr) return ncclSystemError;
    const ncclResult_t result = real(comm, nranks, comm_id, rank);
    if (result == ncclSuccess && comm != nullptr) {
        try {
            probe().remember_comm(*comm, unique_id_hash(comm_id), nranks, rank);
        } catch (...) {
        }
    }
    return result;
}

#if FLASH_NCCL_HAS_INIT_RANK_CONFIG
extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks,
                                                ncclUniqueId comm_id, int rank,
                                                ncclConfig_t* config) {
    safe_api_entry();
    using Function = decltype(&ncclCommInitRankConfig);
    static Function real = resolve_nccl<Function>("ncclCommInitRankConfig");
    if (real == nullptr) return ncclSystemError;
    const ncclResult_t result = real(comm, nranks, comm_id, rank, config);
    if (result == ncclSuccess && comm != nullptr) {
        try {
            probe().remember_comm(*comm, unique_id_hash(comm_id), nranks, rank);
        } catch (...) {
        }
    }
    return result;
}
#endif

extern "C" ncclResult_t ncclGroupStart(void) {
    safe_api_entry();
    using Function = decltype(&ncclGroupStart);
    static Function real = resolve_nccl<Function>("ncclGroupStart");
    if (real == nullptr) return ncclSystemError;
    const ncclResult_t result = real();
    if (result == ncclSuccess) {
        if (thread_context.group_depth == 0) thread_context.group.reset();
        ++thread_context.group_depth;
    }
    return result;
}

extern "C" ncclResult_t ncclGroupEnd(void) {
    safe_api_entry();
    using Function = decltype(&ncclGroupEnd);
    static Function real = resolve_nccl<Function>("ncclGroupEnd");
    if (real == nullptr) return ncclSystemError;
    const ncclResult_t result = real();
    if (thread_context.group_depth > 1) {
        --thread_context.group_depth;
    } else if (thread_context.group_depth == 1) {
        thread_context.group_depth = 0;
        finish_group(result);
    }
    return result;
}

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op,
                                       ncclComm_t comm, cudaStream_t stream) {
    safe_api_entry();
    using Function = decltype(&ncclAllReduce);
    static Function real = resolve_nccl<Function>("ncclAllReduce");
    if (real == nullptr) return ncclSystemError;
    OperationTicket ticket = begin_operation(kAllReduce, count, datatype,
                                             static_cast<int32_t>(op), -1, comm, stream);
    const ncclResult_t result = real(sendbuff, recvbuff, count, datatype, op, comm, stream);
    finish_standalone(ticket, result, stream);
    return result;
}

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclComm_t comm,
                                       cudaStream_t stream) {
    safe_api_entry();
    using Function = decltype(&ncclAllGather);
    static Function real = resolve_nccl<Function>("ncclAllGather");
    if (real == nullptr) return ncclSystemError;
    OperationTicket ticket = begin_operation(kAllGather, count, datatype, -1, -1, comm, stream);
    const ncclResult_t result = real(sendbuff, recvbuff, count, datatype, comm, stream);
    finish_standalone(ticket, result, stream);
    return result;
}

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                                           size_t count, ncclDataType_t datatype,
                                           ncclRedOp_t op, ncclComm_t comm,
                                           cudaStream_t stream) {
    safe_api_entry();
    using Function = decltype(&ncclReduceScatter);
    static Function real = resolve_nccl<Function>("ncclReduceScatter");
    if (real == nullptr) return ncclSystemError;
    OperationTicket ticket = begin_operation(kReduceScatter, count, datatype,
                                             static_cast<int32_t>(op), -1, comm, stream);
    const ncclResult_t result = real(sendbuff, recvbuff, count, datatype, op, comm, stream);
    finish_standalone(ticket, result, stream);
    return result;
}

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
                                       size_t count, ncclDataType_t datatype, int root,
                                       ncclComm_t comm, cudaStream_t stream) {
    safe_api_entry();
    using Function = decltype(&ncclBroadcast);
    static Function real = resolve_nccl<Function>("ncclBroadcast");
    if (real == nullptr) return ncclSystemError;
    OperationTicket ticket = begin_operation(kBroadcast, count, datatype, -1, root,
                                             comm, stream);
    const ncclResult_t result = real(sendbuff, recvbuff, count, datatype, root, comm, stream);
    finish_standalone(ticket, result, stream);
    return result;
}

extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op, int root,
                                    ncclComm_t comm, cudaStream_t stream) {
    safe_api_entry();
    using Function = decltype(&ncclReduce);
    static Function real = resolve_nccl<Function>("ncclReduce");
    if (real == nullptr) return ncclSystemError;
    OperationTicket ticket = begin_operation(kReduce, count, datatype,
                                             static_cast<int32_t>(op), root, comm, stream);
    const ncclResult_t result = real(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    finish_standalone(ticket, result, stream);
    return result;
}

extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count,
                                  ncclDataType_t datatype, int peer, ncclComm_t comm,
                                  cudaStream_t stream) {
    safe_api_entry();
    using Function = decltype(&ncclSend);
    static Function real = resolve_nccl<Function>("ncclSend");
    if (real == nullptr) return ncclSystemError;
    OperationTicket ticket = begin_operation(kSend, count, datatype, -1, peer, comm, stream);
    const ncclResult_t result = real(sendbuff, count, datatype, peer, comm, stream);
    finish_standalone(ticket, result, stream);
    return result;
}

extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                                  int peer, ncclComm_t comm, cudaStream_t stream) {
    safe_api_entry();
    using Function = decltype(&ncclRecv);
    static Function real = resolve_nccl<Function>("ncclRecv");
    if (real == nullptr) return ncclSystemError;
    OperationTicket ticket = begin_operation(kRecv, count, datatype, -1, peer, comm, stream);
    const ncclResult_t result = real(recvbuff, count, datatype, peer, comm, stream);
    finish_standalone(ticket, result, stream);
    return result;
}

