#ifndef PRESSIO_MALLOC_DOMAIN_H
#define PRESSIO_MALLOC_DOMAIN_H
#include <libpressio_ext/cpp/domain.h>
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
    std::string prefix() const override {
        return "malloc";
    }
    std::shared_ptr<pressio_domain> clone() override {
        return shared_from_this();
    }
};
#endif /* end of include guard: PRESSIO_MALLOC_DOMAIN_H */
