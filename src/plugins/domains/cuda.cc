#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_send.h>
#include <libpressio_ext/cpp/registry.h>
#include <cuda_runtime.h>
/**
 * \file
 * \brief domain for cudaMalloc/cudaFree
 */
namespace libpressio { namespace domains { namespace cuda_ns {
struct pressio_cudamalloc_domain: public pressio_domain {
    void* alloc(size_t n) override {
        void* ptr = nullptr;
        auto err = cudaMalloc(&ptr, n);
        if(err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    void free(void* data, size_t) override {
        if(data) {
            cudaFree(data);
            data = nullptr;
        }
    }
    void memcpy(void* dst, void* src, size_t n) override {
        cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice);
    }
    bool equal(pressio_domain const& rhs) const noexcept override {
        return dynamic_cast<pressio_cudamalloc_domain const*>(&rhs) != nullptr;
    }
    std::string const& prefix() const override {
        static const std::string pfx = "cudamalloc";
        return pfx;
    }
    std::shared_ptr<pressio_domain> clone() override {
        return std::make_shared<pressio_cudamalloc_domain>(*this);
    }
};


pressio_register registration(domain_plugins(), "cudamalloc", []{return std::make_shared<pressio_cudamalloc_domain>();});
} } }
