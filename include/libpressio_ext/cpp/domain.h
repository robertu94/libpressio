#ifndef PRESSIO_DOMAIN_H_Z2ALCUZG
#define PRESSIO_DOMAIN_H_Z2ALCUZG
#include <cstddef>
#include <string>
#include <memory>
#include <cstring>
#include <variant>
#include <vector>
#include <map>
#include <any>
#include <type_traits>
#include "std_compat/string_view.h"
#include "registry.h"
#include <libpressio_ext/cpp/names.h>

/**
 * \file
 * \brief C++ Interface for domains of memory
 */


/**
 * domains accept a subset of options that pressio_options do to break the circular dependency
 */
using domain_option  = std::variant<std::monostate,double,uint64_t,int64_t,std::string,std::vector<std::string>,bool,std::any>;
using domain_options  = std::map<std::string, domain_option>;
enum class domain_option_key_status {
    key_set = 0,
    key_exists = 1,
    key_does_not_exist = 2,
};
template <class T>
domain_option_key_status set(domain_options& opts, std::string const& key, T&& value) {
    opts[key] = std::forward<T>(value);
    return domain_option_key_status::key_set;
}
template <class T>
domain_option_key_status set(domain_options& opts, std::string const& name, std::string const& key, T&& value) {
    std::string full_name = libpressio::names::format_name(name, key);
    return set(opts, full_name, std::forward<T>(value));
}

template <class T>
domain_option_key_status get(domain_options const& opts, std::string const& key, T& value) {
    auto it = opts.find(key);
    if (it != opts.end()) {
        if (std::holds_alternative<typename std::decay<T>::type>(it->second)) {
            value = std::get<typename std::decay<T>::type>(it->second);
            return domain_option_key_status::key_set;
        } else {
            return domain_option_key_status::key_exists;
        }

    } else {
        return domain_option_key_status::key_does_not_exist;
    }
}
template <class T>
domain_option_key_status get(domain_options const& opts, std::string const& name, std::string const& key, T& value) {
    std::string prefix_key;
    for (auto path: libpressio::names::search(name)) {
        prefix_key = libpressio::names::format_name(std::string(path), key);
        if(opts.find(prefix_key) != opts.end()) {
            return get(opts, prefix_key, value);
        }
    }
    return get(opts, key, value);
}

template <class Wrapper>
domain_option_key_status set_meta(domain_options& opts, std::string const& name, std::string const& key, Wrapper&& current_value) {
    auto ret = set(opts, name, key, current_value->prefix());
    auto child = current_value->get_options();
    for(auto i: child) {
        opts.insert_or_assign(i.first, i.second);
    }
    return ret;
}
template <class Wrapper, class Registry>
domain_option_key_status get_meta(domain_options& opts, std::string const& name, std::string const& key, Registry const& registry, Wrapper&& current_value) {
    std::string new_plugin;
    auto ret = domain_option_key_status::key_exists;
    if(get(opts, name, key, new_plugin) == domain_option_key_status::key_set) {
        if (new_plugin != current_value->prefix()) {
            auto new_value = registry.build(new_plugin);
            if(new_value) {
                current_value = std::move(new_value);
                ret = domain_option_key_status::key_set;
            } else {
                return domain_option_key_status::key_does_not_exist;
            }
        }
        if(not name.empty()) {
            //TODO this should call back to the set_name function on the caller
            //this does the wrong thing because the parent may name the child in some
            //specific way
            current_value->set_name(name);
        }
    }
    current_value->set_options(opts);
    return ret;
}


std::string to_string(domain_options const& op);
std::string to_string(domain_option const& op);


namespace detail {
    /**
     * search order for keys in domain options
     */
    std::vector<compat::string_view> search(compat::string_view const& value);

    /**
     * helper to assign a value to a domain option if and only if the type matches
     */
    template <class T>
    struct maybe_assign {
        template <class V>
        typename std::enable_if<
            std::is_same<
                typename std::decay<T>::type,
                typename std::decay<V>::type>
            ::value,
            domain_option_key_status 
        >::type
        operator()(V&& rhs) {
            lhs = rhs;
            return domain_option_key_status::key_set;
        }
        template <class V>
        typename std::enable_if<
            !std::is_same<
                typename std::decay<T>::type,
                typename std::decay<V>::type>
            ::value,
            domain_option_key_status 
        >::type
        operator()(V&& rhs) {
            (void) rhs;
            return domain_option_key_status::key_exists;
        }
        T&& lhs;
    };
}

template <class T>
domain_option_key_status get(domain_options const& opts, std::string const& key, T&& value) {
    auto it = opts.find(key);
    if(it != opts.end()) {
        return std::visit(detail::maybe_assign<T>{value}, it->second);
    } else {
        return domain_option_key_status::key_does_not_exist;
    }
}
template <class T>
domain_option_key_status get(domain_options const& opts, std::string const& name, std::string const& key, T&& value) {
    std::string prefix_key;
    if(!name.empty()) {
        for (auto path : detail::search(name)) {
            prefix_key = '/' + std::string(path) + ':' + key;
            if (opts.find(prefix_key) == opts.end()) {
                return get(opts, prefix_key, value);
            }
        }
    }
    return get(opts, key, value);
}

/**
 * represents a malloc/free pair
 */
struct pressio_domain {
    public:
    /**
     * set the options for the domain.  May in the future collect metrics or allow callbacks
     * \param[in] opts the options to set
     * \returns 0 on ok, >0 on error, <0 on warning
     */
    int set_options(domain_options const& opts) {
        return set_options_impl(opts);
    }
    /**
     * get the options for the domain.  May in the future collect metrics or allow callbacks
     * \returns the options set on the domain
     */
    domain_options get_options() const {
        return get_options_impl();
    }
    /**
     * get the documentation for the domain.  May in the future collect metrics or allow callbacks
     * \returns the options set on the domain
     */
    domain_options get_documentation() const {
        return get_documentation_impl();
    }
    /**
     * get the configuration for the domain.  May in the future collect metrics or allow callbacks
     * \returns the options set on the domain
     */
    domain_options get_configuration() const {
        return get_configuration_impl();
    }
    /**
     * allocate memory in the domain
     *
     * an implementation MUST provide at least this many bytes writable from the devices that are accessible from this domain
     * an implementation MUST ensure that all allocated regions do not overlap
     * an implementation SHOULD avoid wastefully large over-allocations
     * an implementation MAY throw an exception if allocation fails
     *
     * \param[in] n minimum amount of memory in bytes to allocate
     */
    virtual void* alloc(size_t n)=0;
    /**
     * frees memory associated from the domain
     *
     * users MUST provide the same capacity used to the call to alloc
     * users MUST NOT utilize a pointer after it has been passed to free until it is returned from a subsequent alloc operation
     * an implementation SHOULD return the underlying allocation to the device
     *
     * \param[in] data the pointer to the data to be freed previously allocated by alloc
     * \param[in] capacity the amount of the memory to return to the allocator
     */
    virtual void free(void* data, size_t capacity)=0;
    /**
     * moves memory within a domain
     *
     * users MUST provided a src and dst pointer from a domain that is equal to this domain
     * users MUST provide a capacity that does not exceed the capacity of the smaller of the src or dst
     *
     * \param[out] dst where to copy the data to
     * \param[in] src where to copy the data from
     * \param[in] capacity the number of bytes to copy
     */
    virtual void memcpy(void* dst, void* src, size_t capacity)=0;
    /**
     * name for this particular domain
     */
    void set_name(std::string const&);
    /**
     * name for this particular domain
     */
    /**
     * name for this particular domain
     */
    std::string const& get_name() const {
        return name;
    }
    /**
     * a unique identifier for all domains that can memcpy from this domain
     *
     * it MUST be pass-able to domain_plugins().build(x) to construct an allocator
     * that can allocate in domains that share this id
     *
     * the prefix and the domain_id MAY be different
     */
    virtual std::string const& domain_id() const {
        return prefix();
    }
    /**
     * a unique identifier for all domains that compare equal to this domain
     *
     * it MUST be pass-able to domain_plugins().build(x) to construct an allocator
     * that can free in domains that share this id
     *
     * the prefix and the domain_id MAY be different
     */
    virtual std::string const& prefix() const=0;
    /**
     * \returns true if and only if pointers allocated with this can be freed by rhs or visa-versa
     */
    bool operator==(pressio_domain const& rhs) const noexcept {
        return equal(rhs);
    }
    /**
     * clones a domain
     *
     * an implementation MUST preform allocations on the same device
     * an implementation SHOULD return a domain that compares equal to this domain
     */
    virtual std::shared_ptr<pressio_domain> clone()=0;

    protected:
    /**
     * set the options for the domain.  Intended to be overwritten by implementing classes when needed
     * implementations SHOULD ignore settings they do not recognize 
     * implementations SHOULD only return errors for invalid configurations
     * implementations SHOULD only return warnings for configurations that the implementation knows to be suboptimal
     * \param[in] opts the options to set
     * \returns 0 on ok, >0 on error, <0 on warning
     */
    virtual int set_options_impl(domain_options const& opts) {
        (void)opts;
        return 0;
    }

    /**
     * get the options for the domain.  Intended to be overwritten by implementing classes when needed
     * \returns the options set on the domain
     */
    virtual domain_options get_options_impl() const {
        return {};
    }
    /**
     * get the documentation for the domain.  Intended to be overwritten by implementing classes when needed
     * \returns the options set on the domain
     */
    virtual domain_options get_documentation_impl() const {
        return {};
    }
    /**
     * get the configuration for the domain.  Intended to be overwritten by implementing classes when needed
     * \returns the options set on the domain
     */
    virtual domain_options get_configuration_impl() const {
        return {};
    }
    virtual void set_name_impl(std::string const&) {
    }
    /**
     * test if another domain may free memory from this domain
     */
    virtual bool equal(pressio_domain const&) const noexcept=0;

    /**
     * destroy
     */
    virtual ~pressio_domain()=default;
    pressio_domain()=default;

    private:
    std::string name = "";
};

/**
 * helper function that test if two domains are accessible from each other (i.e. can they use
 * each others memcpy function to copy from one to another)
 */
bool is_accessible(pressio_domain const& lhs, pressio_domain const& rhs);

pressio_registry<std::shared_ptr<pressio_domain>>& domain_plugins();

/**
 * returns a domain that matches the desired characteristics
 */
std::shared_ptr<pressio_domain> find_domain(domain_options const& characteristics);


#endif /* end of include guard: PRESSIO_DOMAIN_H_Z2ALCUZG */
