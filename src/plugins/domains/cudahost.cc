#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/registry.h>
#include <cuda_runtime.h>
/**
 * \file
 * \brief domain for cudamallochost/cudaFree
 */
namespace libpressio {namespace domains { namespace cudahost_ns {
struct pressio_cudamallochost_domain: public pressio_domain, std::enable_shared_from_this<pressio_cudamallochost_domain> {
    void* alloc(size_t n) override {
        void* ptr = nullptr;
        auto err = cudaMallocHost(&ptr, n);
        if(err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    void free(void* data, size_t) override {
        if(data) {
            cudaFreeHost(data);
            data = nullptr;
        }
    }

    domain_options get_configuration_impl() const override {
        domain_options opts;
        set(opts, get_name(), "domains:accessible", std::vector<std::string>{"malloc"});
        return opts;
    }

    void memcpy(void* dst, void* src, size_t n) override {
        cudaMemcpy(dst, src, n, cudaMemcpyHostToHost);
    }
    bool equal(pressio_domain const& rhs) const noexcept override {
        return dynamic_cast<pressio_cudamallochost_domain const*>(&rhs) != nullptr;
    }
    std::string const& prefix() const override {
        static const std::string pfx = "cudamallochost"; 
        return pfx;
    }
    std::shared_ptr<pressio_domain> clone() override {
        return shared_from_this();
    }
};


pressio_register registration(domain_plugins(), "cudamallochost", []{return std::make_shared<pressio_cudamallochost_domain>();});
}}}
