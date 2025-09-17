#include <libpressio_ext/cpp/data.h>
#include "cuda_domain_send.h"
#include <cuda_runtime.h>

namespace libpressio { namespace domains {
void pressio_domain_send_host_to_device::send(pressio_data& dst, pressio_data const& src) const {
    auto err = cudaMemcpy(dst.data(), src.data(), dst.size_in_bytes(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
void pressio_domain_send_device_to_host::send(pressio_data& dst, pressio_data const& src) const {
    auto err = cudaMemcpy(dst.data(), src.data(), dst.size_in_bytes(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
} }
