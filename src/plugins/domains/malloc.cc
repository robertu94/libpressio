#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_send.h>
#include <libpressio_ext/cpp/registry.h>
/**
 * \file
 * \brief domain for std::malloc/std::free
 */
namespace libpressio { namespace domains { namespace malloc_ns {
struct pressio_malloc_domain: public pressio_domain, std::enable_shared_from_this<pressio_malloc_domain> {
    void* alloc(size_t n) override {
        return std::malloc(n);
    }
    void free(void* data, size_t) override {
        if(data) std::free(data);
    }
    void memcpy(void* dst, void* src, size_t n) override {
        std::memcpy(dst, src, n);
    }
    bool equal(pressio_domain const& rhs) const noexcept override {
        return dynamic_cast<pressio_malloc_domain const*>(&rhs) != nullptr;
    }
    std::string const& prefix() const override {
        static std::string const& pfx = "malloc";
        return pfx;
    }
    std::shared_ptr<pressio_domain> clone() override {
        return shared_from_this();
    }
};
pressio_register registration(domain_plugins(), "malloc", []{return std::make_shared<pressio_malloc_domain>();});
}}}
