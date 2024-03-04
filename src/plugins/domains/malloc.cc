#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_send.h>
#include <libpressio_ext/cpp/registry.h>
/**
 * \file
 * \brief domain for std::malloc/std::free
 */
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


pressio_register malloc_register(domain_plugins(), "malloc", []{return std::make_shared<pressio_malloc_domain>();});

struct pressio_domain_send_host_to_host: public pressio_domain_send {
    void send(pressio_data& dst, pressio_data const& src) const override {
        memcpy(dst.data(), src.data(), dst.size_in_bytes());
    }
};
