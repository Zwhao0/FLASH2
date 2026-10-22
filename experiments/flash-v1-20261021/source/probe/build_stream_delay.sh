#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR:-${CUDA_ROOT}/include}"
NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-/usr/include}"
CXX="${CXX:-g++}"

if [[ -z "${CUDA_LIB_DIR:-}" ]]; then
    if [[ -e "${CUDA_ROOT}/lib64/libcudart.so" ]]; then
        CUDA_LIB_DIR="${CUDA_ROOT}/lib64"
    elif [[ -e "${CUDA_ROOT}/targets/x86_64-linux/lib/libcudart.so" ]]; then
        CUDA_LIB_DIR="${CUDA_ROOT}/targets/x86_64-linux/lib"
    else
        echo "libcudart.so not found under ${CUDA_ROOT}; set CUDA_LIB_DIR" >&2
        exit 1
    fi
fi

if [[ ! -f "${CUDA_INCLUDE_DIR}/cuda_runtime_api.h" ]]; then
    echo "cuda_runtime_api.h not found in ${CUDA_INCLUDE_DIR}; set CUDA_INCLUDE_DIR" >&2
    exit 1
fi
if [[ ! -f "${NCCL_INCLUDE_DIR}/nccl.h" ]]; then
    echo "nccl.h not found in ${NCCL_INCLUDE_DIR}; set NCCL_INCLUDE_DIR" >&2
    exit 1
fi

"${CXX}" \
    -std=c++17 -O2 -fPIC -shared -pthread \
    -I"${CUDA_INCLUDE_DIR}" -I"${NCCL_INCLUDE_DIR}" \
    "${SCRIPT_DIR}/stream_delay.cpp" \
    -L"${CUDA_LIB_DIR}" -Wl,-rpath,"${CUDA_LIB_DIR}" \
    -lcudart -ldl -o "${SCRIPT_DIR}/libflash_stream_delay.so"

echo "Built ${SCRIPT_DIR}/libflash_stream_delay.so"
