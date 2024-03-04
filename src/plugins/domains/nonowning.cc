#include "libpressio_ext/cpp/domain.h"

/**
 * \file
 * \brief domain that does not own pointers, and thus cannot allocate of free them
 */
struct pressio_nonowning_domain: public pressio_domain, std::enable_shared_from_this<pressio_nonowning_domain> {
    domain_options get_options_impl() const override {
        domain_options opts;
        set(opts, get_name(), "nonowning:domain_id", prefix_str);
        return opts;
    }
    int set_options_impl(domain_options const& opts) override {
        get(opts, get_name(), "nonowning:domain_id", prefix_str);
        return 0;
    }
    void* alloc(size_t) override {
        throw std::bad_alloc();
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
    std::string const& prefix() const override {
        static std::string const pfx = "nonowning";
        return pfx;
    }
    std::string const& domain_id() const override {
        return prefix_str;
    }
    std::shared_ptr<pressio_domain> clone() override {
        return shared_from_this();
    }
    std::string prefix_str;
};
pressio_register nonowning_domain(domain_plugins(), "nonowning", []{return std::make_shared<pressio_nonowning_domain>();});

