#include "libpressio_ext/cpp/domain.h"

/**
 * \file
 * \brief domain that does not own pointers, and thus cannot allocate of free them
 */
namespace libpressio { namespace domains { namespace nonowning_ns {
struct pressio_nonowning_domain: public pressio_domain, std::enable_shared_from_this<pressio_nonowning_domain> {
    domain_options get_configuration_impl() const override {
        domain_options opts;
        set(opts, get_name(), "domains:accessible", accessible_domains);
        return opts;
    }
    domain_options get_options_impl() const override {
        domain_options opts;
        set(opts, get_name(), "nonowning:domain_id", prefix_str);
        return opts;
    }
    int set_options_impl(domain_options const& opts) override {
        if(get(opts, get_name(), "nonowning:domain_id", prefix_str) == domain_option_key_status::key_set) {
            std::vector<std::string> accessible;
            auto base_domain = domain_plugins().build(prefix_str);
            auto base_configuration = base_domain->get_configuration();
            accessible_domains.clear();
            if(get(base_configuration, "domains:accessible", accessible) == domain_option_key_status::key_set) {
                accessible_domains.reserve(1+accessible.size());
                accessible_domains.emplace_back(prefix_str);
                accessible_domains.insert(accessible_domains.end(),
                        std::make_move_iterator(accessible.begin()),
                        std::make_move_iterator(accessible.end()));
            } else {
                accessible_domains.reserve(1);
                accessible_domains.emplace_back(prefix_str);
            }
        }
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
        return domain_plugins().build(domain_id());
    }
    std::string prefix_str = "malloc";
    std::vector<std::string> accessible_domains{prefix_str};
};
pressio_register registration(domain_plugins(), "nonowning", []{return std::make_shared<pressio_nonowning_domain>();});

}}}
