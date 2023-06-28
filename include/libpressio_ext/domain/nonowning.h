#ifndef PRESSIO_NONOWNING_DOMAIN_H
#define PRESSIO_NONOWNING_DOMAIN_H

#include "libpressio_ext/cpp/domain.h"

struct pressio_nonowning_domain: public pressio_domain, std::enable_shared_from_this<pressio_nonowning_domain> {
    void* alloc(size_t) override {
        return nullptr;
    }
    void free(void*, size_t) override {
        return;
    }
    void memcpy(void*, void*, size_t) override {
        return;
    }
    bool equal(pressio_domain const& rhs) const noexcept override {
        return this == &rhs;
    }
    std::string prefix() const override {
        std::stringstream ss;
        ss << "nonowning:" << intptr_t(this);
        return "nonowning";
    }
    std::shared_ptr<pressio_domain> clone() override {
        return shared_from_this();
    }
};

#endif /* end of include guard: PRESSIO_NONOWNING_DOMAIN_H */
