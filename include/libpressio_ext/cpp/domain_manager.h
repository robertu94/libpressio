#ifndef LIBPRESSIO_DOMAIN_MANAGER_H
#define LIBPRESSIO_DOMAIN_MANAGER_H
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_send.h>
#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/registry.h>
#include <libpressio_ext/cpp/names.h>

namespace libpressio { namespace domains_metrics {
    using domains::domain_options;
    using domains::set;
/**
 * callback class for the domain manager
 */
struct pressio_domain_manager_metrics_plugin {
    pressio_domain_manager_metrics_plugin()=default;
    virtual ~pressio_domain_manager_metrics_plugin()=default;
    /**
     * called before memory is allocated from a domain
     */
    virtual void alloc_begin(std::shared_ptr<domains::pressio_domain> const& domain, pressio_dtype dtype, std::vector<size_t> const& dims) { (void)domain;(void)dtype; (void)dims;};
    /**
     * called after memory is allocated from a domain
     */
    virtual void alloc_end(std::shared_ptr<domains::pressio_domain> const& domain, pressio_dtype dtype, std::vector<size_t> const& dims) { (void)domain; (void)dtype; (void)dims;};
    /**
     * called before memory is viewed from a domain
     */
    virtual void view_begin(std::shared_ptr<domains::pressio_domain> const& dst, pressio_data const& src) { (void)src; (void)dst;};
    /**
     * called after memory is viewed from a domain
     */
    virtual void view_end(std::shared_ptr<domains::pressio_domain> const& dst, pressio_data const& src) { (void)src; (void)dst; };
    /**
     * called before data is sent from a domain
     */
    virtual void send_begin(pressio_data const& dst, pressio_data const& src) { (void)src; (void)dst; };
    /**
     * called after data is sent from a domain
     */
    virtual void send_end(pressio_data const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called before data is requested to be readable with data
     */
    virtual void make_readable_begin(pressio_data const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called after data is requested to be readable with data
     */
    virtual void make_readable_end(pressio_data const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called before data is requested to be writeable with data
     */
    virtual void make_writeable_begin(std::shared_ptr<domains::pressio_domain> const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called after data is requested to be writeable with data
     */
    virtual void make_writeable_end(std::shared_ptr<domains::pressio_domain> const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called before data is requested to be readable on a domain
     */
    virtual void make_readable_domain_begin(std::shared_ptr<domains::pressio_domain> const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called after data is requested to be readable on a domain
     */
    virtual void make_readable_domain_end(std::shared_ptr<domains::pressio_domain> const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called before data is copied
     */
    virtual void copy_to_begin(pressio_data const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called after data is copied
     */
    virtual void copy_to_end(pressio_data const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * set options on this domain metrics
     */
    virtual int set_options(domains::domain_options const&) { return 0; }
    /**
     * get options on this domain metrics
     */
    virtual domains::domain_options get_options() const { return domains::domain_options{}; }
    /**
     * return the metrics from this domain
     */
    virtual domains::domain_options get_metrics_results() { return domains::domain_options{}; }
    /**
     * clone this domain metrics object
     */
    virtual std::unique_ptr<pressio_domain_manager_metrics_plugin> clone() const {
        return std::make_unique<pressio_domain_manager_metrics_plugin>();
    }
    /*
     * id of the module
     */
    virtual const char* prefix() const {
        return "noop";
    }

    /**
     * return the name of the module
     */
    std::string const& get_name() {
        return name;
    }
    /**
     * set the name of the module
     */
    void set_name(std::string const& new_name) {
        set_name_impl(new_name);
        name = new_name;
    }
    /**
     * callback when the name is updated
     */
    virtual void set_name_impl(std::string const&) {
    }
    private:
    std::string name;
};
}}

/**
 * the registry for metrics plugins
 */
libpressio::pressio_registry<std::unique_ptr<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin>>& domain_metrics_plugins();

/**
 * pointer manager object for domain manager metrics
 */
struct pressio_domain_manager_metrics {
    pressio_domain_manager_metrics(libpressio::domains_metrics::pressio_domain_manager_metrics_plugin const& rhs): plg(rhs.clone()) {}
    pressio_domain_manager_metrics(libpressio::domains_metrics::pressio_domain_manager_metrics_plugin & rhs): plg(std::make_unique<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin>(std::move(rhs))) {}

    pressio_domain_manager_metrics(): plg(std::make_unique<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin>()) {}
    pressio_domain_manager_metrics(std::unique_ptr<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin> && rhs): plg(std::move(rhs)) {}
    pressio_domain_manager_metrics(pressio_domain_manager_metrics const& rhs): plg(rhs.plg->clone()) {}
    pressio_domain_manager_metrics(pressio_domain_manager_metrics && rhs) noexcept: plg(std::exchange(rhs.plg, {})) {}
    pressio_domain_manager_metrics& operator=(pressio_domain_manager_metrics const& rhs) noexcept  {
        if(&rhs == this) return *this;
        plg = rhs.plg->clone();
        return *this;
    }
    pressio_domain_manager_metrics& operator=(pressio_domain_manager_metrics && rhs) noexcept  {
        if(&rhs == this) return *this;
        plg = std::exchange(rhs.plg, {});
        return *this;
    }

    libpressio::domains_metrics::pressio_domain_manager_metrics_plugin& operator*() { return *plg; }
    libpressio::domains_metrics::pressio_domain_manager_metrics_plugin* operator->() { return plg.operator->(); }
    libpressio::domains_metrics::pressio_domain_manager_metrics_plugin const& operator*() const { return *plg; }
    libpressio::domains_metrics::pressio_domain_manager_metrics_plugin const* operator->() const { return plg.operator->(); }
    operator bool() const { return static_cast<bool>(plg); }
    std::unique_ptr<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin> plg;
};

namespace libpressio { namespace domains {
/**
 * manager to send data
 */
struct pressio_domain_manager {
    public:
    virtual ~pressio_domain_manager()=default;
    /**
     * send data from one non-accessible domain to another
     */
    void send(pressio_data& dst, pressio_data const& src) {
        metrics->send_begin(dst, src);
        send_impl(dst, src);
        metrics->send_end(dst, src);
    }

    /**
     * make data readable using the provided memory if possible
     */
    template <class T>
    pressio_data make_readable(pressio_data&& dst, T&& src) {
        metrics->make_readable_begin(dst, src);
        auto ret = make_readable_impl(std::move(dst), std::forward<T>(src));
        metrics->make_readable_end(dst, src);
        return ret;
    }
    /**
     * make data readable using the provided domain
     */
    template <class T>
    pressio_data make_readable(std::shared_ptr<pressio_domain>&& dst, T&& src) {
        metrics->make_readable_domain_begin(dst, src);
        auto ret = make_readable_impl(std::move(dst), std::forward<T>(src));
        metrics->make_readable_domain_end(dst, src);
        return ret;
    }

    /**
     * copy data to the specified memory, or move if both pointers are in the same domain
     */
    template <class T>
    pressio_data copy_to(pressio_data&& dst, T&& src) {
        metrics->copy_to_begin(dst, src);
        auto ret = copy_to_impl(dst, std::forward<T>(src));
        metrics->copy_to_end(dst, src);
        return ret;
    }

    /**
     * copy data to the specified memory, or move if both pointers are in the same domain
     */
    template <class T>
    pressio_data copy_to(std::shared_ptr<pressio_domain>&& dst, T&& src) {
        auto ret = copy_to_impl(std::move(dst), std::forward<T>(src));
        return ret;
    }

    /**
     * make the buffer writable in the specified domain
     */
    template <class T>
    pressio_data make_writeable(std::shared_ptr<pressio_domain>&& dst, T&& data) {
        metrics->make_writeable_begin(dst, data);
        auto ret = make_writeable_impl(std::move(dst), std::forward<T>(data));
        metrics->make_writeable_end(dst, data);
        return ret;
    }

    /**
     * set the metrics for this domain
     */
    void set_metrics(pressio_domain_manager_metrics&& metrics) {
        this->metrics = std::move(metrics);
    }
    /**
     * set the options on the domain manager
     */
    int set_options(domain_options&& opts) {
        get_meta(opts, name, "domain:metrics", domain_metrics_plugins(), this->metrics);
        return this->metrics->set_options(opts);
    }

    void set_name(std::string const& new_name) {
        name = new_name;
        if(new_name.empty()) {
            this->metrics->set_name(new_name);
        } else {
            this->metrics->set_name(new_name + "/metrics");
        }
    }
    /**
     * get the options on the domain manager
     */
    domain_options get_options() const {
        return this->metrics->get_options();
    }
    /**
     * get the metrics results form the domain manager
     */
    domain_options get_metrics_results() {
        return metrics->get_metrics_results();
    }

    protected:
    virtual void send_impl(pressio_data& dst, pressio_data const& src) {
        auto method = src.domain()->domain_id() + ">" +  dst.domain()->domain_id();
        auto sender = domain_send_plugins().build(method);
        if(sender) {
           sender->send(dst, src);
        } else {
            throw std::runtime_error("no viable send method for " + method);
        }
    }


    virtual pressio_data make_readable_impl(pressio_data&& dst, pressio_data const& src) {
        if(is_accessible(*dst.domain(), *src.domain())) {
            metrics->view_begin(dst.domain(), src);
            dst = pressio_data::nonowning(src);
            metrics->view_end(dst.domain(), src);
        } else {
            if(src.has_data()) {
                if (!dst.has_data()) {
                    auto domain = dst.domain();
                    metrics->alloc_begin(domain, src.dtype(), src.dimensions());
                    dst = pressio_data::owning(src.dtype(), src.dimensions(), domain);
                    metrics->alloc_end(domain, src.dtype(), src.dimensions());
                }
                send(dst, src);
            } else {
                dst = pressio_data::empty(src.dtype(), src.dimensions(), dst.domain());
            }
        }
        return std::move(dst);
    }
    virtual pressio_data make_readable_impl(pressio_data&& dst, pressio_data&& src) {
        return make_readable_impl(std::move(dst), src);
    }
    virtual pressio_data make_readable_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data const& src) {
        if(is_accessible(*dst, *src.domain())) {
            metrics->view_begin(dst, src);
            pressio_data out(pressio_data::nonowning(src));
            metrics->view_end(dst, src);
            return out;
        } else {
            if(src.has_data()) {
                metrics->alloc_begin(dst, src.dtype(), src.dimensions());
                pressio_data out(pressio_data::owning(src.dtype(), src.dimensions(), dst));
                metrics->alloc_end(dst, src.dtype(), src.dimensions());
                send(out, src);
                return out;
            } else {
                return pressio_data::empty(src.dtype(), src.dimensions(), std::move(dst));
            }
        }
    }
    virtual pressio_data make_readable_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data&& src) {
        return make_readable_impl(std::move(dst), src);
    }

    virtual pressio_data copy_to_impl(pressio_data&& dst, pressio_data &&src) {
        if(!src.has_data()) throw std::runtime_error("cannot send from a source that is unallocated");
        if(is_accessible(*dst.domain(), *src.domain())) {
            return std::move(src);
        } else {
            if(!dst.has_data()) {
                auto const& domain = dst.domain();
                metrics->alloc_begin(domain, src.dtype(), src.dimensions());
                dst = pressio_data::owning(src.dtype(), src.dimensions(), domain);
                metrics->alloc_end(domain, src.dtype(), src.dimensions());
            }
            send(dst, src);
            return std::move(dst);
        }
    }
    virtual pressio_data copy_to_impl(pressio_data&& dst, pressio_data const&src) {
        if(!src.has_data()) throw std::runtime_error("cannot send from a source that is unallocated");
        if(is_accessible(*dst.domain(), *src.domain())) {
            dst = src;
            return dst;
        } else {
            if(!dst.has_data()) {
                auto const& domain = dst.domain();
                metrics->alloc_begin(domain, src.dtype(), src.dimensions());
                dst = pressio_data::owning(src.dtype(), src.dimensions(), domain);
                metrics->alloc_end(domain, src.dtype(), src.dimensions());
            }
            send(dst, src);
            return dst;
        }
    }
    virtual pressio_data copy_to_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data &&src) {
        if(!src.has_data()) throw std::runtime_error("cannot send from a source that is unallocated");
        if(is_accessible(*dst, *src.domain())) {
            return std::move(src);
        } else {
            metrics->alloc_begin(dst, src.dtype(), src.dimensions());
            auto out = pressio_data::owning(src.dtype(), src.dimensions(), dst);
            metrics->alloc_end(dst, src.dtype(), src.dimensions());
            send(out, src);
            return out;
        }
    }
    virtual pressio_data copy_to_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data const&src) {
        if(!src.has_data()) throw std::runtime_error("cannot send from a source that is unallocated");
        if(is_accessible(*dst, *src.domain())) {
            metrics->alloc_begin(dst, src.dtype(), src.dimensions());
            pressio_data out = pressio_data::owning(src.dtype(), src.dimensions(), std::move(dst));
            metrics->alloc_end(dst, src.dtype(), src.dimensions());
            out = src;
            return out;
        } else {
            metrics->alloc_begin(dst, src.dtype(), src.dimensions());
            auto out = pressio_data::owning(src.dtype(), src.dimensions(), dst);
            metrics->alloc_end(dst, src.dtype(), src.dimensions());
            send(out, src);
            return out;
        }
    }

    virtual pressio_data make_writeable_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data const& src) {
        metrics->alloc_begin(dst, src.dtype(), src.dimensions());
        pressio_data out = pressio_data::owning(src.dtype(), src.dimensions(), std::move(dst));
        metrics->alloc_end(dst, src.dtype(), src.dimensions());
        return out;
    }
    virtual pressio_data make_writeable_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data && src) {
        if(is_accessible(*dst, *src.domain()) && src.has_data()) {
            metrics->view_begin(dst, src);
            pressio_data out(std::move(src));
            metrics->view_end(dst, src);
            return out;
        } else {
            metrics->alloc_begin(dst, src.dtype(), src.dimensions());
            pressio_data out = pressio_data::owning(src.dtype(), src.dimensions(), std::move(dst));
            metrics->alloc_end(dst, src.dtype(), src.dimensions());
            return out;
        }
    }

    std::string name;
    pressio_domain_manager_metrics metrics;
};

} }
/**
 * returns a reference to the default global domain manager
 */
libpressio::domains::pressio_domain_manager& domain_manager();

#endif /* end of include guard: LIBPRESSIO_DOMAIN_MANAGER_H */
