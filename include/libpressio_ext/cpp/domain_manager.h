#ifndef LIBPRESSIO_DOMAIN_MANAGER_H
#define LIBPRESSIO_DOMAIN_MANAGER_H
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_send.h>
#include <libpressio_ext/cpp/data.h>

/**
 * callback class for the domain manager
 */
struct pressio_domain_manager_metrics_plugin {
    pressio_domain_manager_metrics_plugin()=default;
    virtual ~pressio_domain_manager_metrics_plugin()=default;
    /**
     * called before memory is allocated from a domain
     */
    virtual void alloc_begin(std::shared_ptr<pressio_domain> const& domain) { (void)domain;};
    /**
     * called after memory is allocated from a domain
     */
    virtual void alloc_end(std::shared_ptr<pressio_domain> const& domain) { (void)domain;};
    /**
     * called before memory is viewed from a domain
     */
    virtual void view_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) { (void)src; (void)dst;};
    /**
     * called after memory is viewed from a domain
     */
    virtual void view_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) { (void)src; (void)dst; };
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
     * called before data is requested to be readable on a domain
     */
    virtual void make_readable_domain_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) {(void)src; (void)dst;};
    /**
     * called after data is requested to be readable on a domain
     */
    virtual void make_readable_domain_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) {(void)src; (void)dst;};
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
    virtual int set_options(domain_options const&) { return 0; }
    /**
     * get options on this domain metrics
     */
    virtual domain_options get_options() const { return domain_options{}; }
    /**
     * return the metrics from this domain
     */
    virtual domain_options get_metrics_results() { return domain_options{}; }
    /**
     * clone this domain metrics object
     */
    virtual std::unique_ptr<pressio_domain_manager_metrics_plugin> clone() const {
        return std::make_unique<pressio_domain_manager_metrics_plugin>();
    }
};
/**
 * pointer manager object for domain manager metrics
 */
struct pressio_domain_manager_metrics {
    pressio_domain_manager_metrics(pressio_domain_manager_metrics_plugin const& rhs): plg(rhs.clone()) {}
    pressio_domain_manager_metrics(pressio_domain_manager_metrics_plugin & rhs): plg(std::make_unique<pressio_domain_manager_metrics_plugin>(std::move(rhs))) {}

    pressio_domain_manager_metrics(): plg(std::make_unique<pressio_domain_manager_metrics_plugin>()) {}
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

    pressio_domain_manager_metrics_plugin& operator*() { return *plg; }
    pressio_domain_manager_metrics_plugin* operator->() { return plg.operator->(); }
    pressio_domain_manager_metrics_plugin const& operator*() const { return *plg; }
    pressio_domain_manager_metrics_plugin const* operator->() const { return plg.operator->(); }
    operator bool() const { return static_cast<bool>(plg); }
    std::unique_ptr<pressio_domain_manager_metrics_plugin> plg;
};

/**
 * manager to send data
 */
struct pressio_domain_manager {
    public:
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
     * set the metrics for this domain
     */
    void set_metrics(pressio_domain_manager_metrics&& metrics) {
        this->metrics = std::move(metrics);
    }
    /**
     * set the options on the domain manager
     */
    int set_options(domain_options&& opts) {
        return this->metrics->set_options(opts);
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
            dst = pressio_data::nonowning(src.dtype(), src.data(), src.dimensions());
            metrics->view_end(dst.domain(), src);
        } else {
            if (!dst.has_data()) {
                auto domain = dst.domain();
                metrics->alloc_begin(domain);
                dst = pressio_data::owning(src.dtype(), src.dimensions(), domain);
                metrics->alloc_end(domain);
            }
            send(dst, src);
        }
        return std::move(dst);
    }
    virtual pressio_data make_readable_impl(pressio_data&& dst, pressio_data&& src) {
        return make_readable_impl(std::move(dst), src);
    }
    virtual pressio_data make_readable_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data const& src) {
        pressio_data out;
        if(is_accessible(*dst, *src.domain())) {
            metrics->view_begin(dst, src);
            out = pressio_data::nonowning(src.dtype(), src.data(), src.dimensions());
            metrics->view_end(dst, src);
        } else {
            metrics->alloc_begin(dst);
            out = pressio_data::owning(src.dtype(), src.dimensions(), dst);
            metrics->alloc_end(dst);
            send(out, src);
        }
        return out;
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
                metrics->alloc_begin(domain);
                dst = pressio_data::owning(src.dtype(), src.dimensions(), domain);
                metrics->alloc_end(domain);
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
                metrics->alloc_begin(domain);
                dst = pressio_data::owning(src.dtype(), src.dimensions(), domain);
                metrics->alloc_end(domain);
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
            metrics->alloc_begin(dst);
            auto out = pressio_data::owning(src.dtype(), src.dimensions(), dst);
            metrics->alloc_end(dst);
            send(out, src);
            return out;
        }
    }
    virtual pressio_data copy_to_impl(std::shared_ptr<pressio_domain>&& dst, pressio_data const&src) {
        if(!src.has_data()) throw std::runtime_error("cannot send from a source that is unallocated");
        if(is_accessible(*dst, *src.domain())) {
            metrics->alloc_begin(dst);
            pressio_data out = pressio_data::owning(src.dtype(), src.dimensions(), std::move(dst));
            metrics->alloc_end(dst);
            out = src;
            return out;
        } else {
            metrics->alloc_begin(dst);
            auto out = pressio_data::owning(src.dtype(), src.dimensions(), dst);
            metrics->alloc_end(dst);
            send(out, src);
            return out;
        }
    }
    pressio_domain_manager_metrics metrics;
};

/**
 * returns a reference to the default global domain manager
 */
pressio_domain_manager& domain_manager();

#endif /* end of include guard: LIBPRESSIO_DOMAIN_MANAGER_H */
