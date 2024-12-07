#include "user.h"
#include <cstddef>
#include <cstring>
void* pressio_libc_alloc_fn(size_t n, void*) {
    return malloc(n);
}
void* pressio_noop_alloc_fn(size_t n, void*) {
    return nullptr;
}
void pressio_libc_memcpy_fn(void* dst, void* src, size_t n, void*) {
    memcpy(dst, src, n);
}
#if LIBPRESSIO_HAS_CUDA
#include <cuda_runtime.h>
extern "C" {
    void* pressio_cuda_alloc_fn(size_t n, void*) {
        void* ptr;
        cudaMalloc(&ptr, n);
        return ptr;
    }
    void* pressio_cudahost_alloc_fn(size_t n, void*) {
        void* ptr;
        cudaMallocHost(&ptr, n);
        return ptr;
    }
    void  pressio_cuda_memcpy_fn(void* dst, void* src, size_t n, void*) {
        cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice);
    }
}
#endif
