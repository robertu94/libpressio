#include <iostream>
#include <chrono>
#include <sstream>
#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/printers.h>
#include <libpressio_ext/cpp/domain_manager.h>

namespace libpressio { namespace domain_metrics { namespace print {

std::string prefix_or_name(pressio_domain const& d) {
    std::stringstream ss;
    if(d.get_name().empty()) {
        if(d.domain_id() == d.prefix()) {
            ss << d.domain_id();
        } else {
            ss << d.prefix() << '(' << d.domain_id() << ')';
        }
    } else {
        if(d.get_name() == d.prefix()) {
            ss << d.get_name();
        } else {
            ss << d.get_name() << '(' << d.domain_id() << ')';
        }
    }
    return ss.str();
}
std::string prefix_or_name(std::shared_ptr<const pressio_domain> const& d) {
    if(d == nullptr) return "{moved}";
    else return prefix_or_name(*d);
}

struct print_plugin: public pressio_domain_manager_metrics_plugin {
    print_plugin()=default;
    virtual ~print_plugin()=default;

    struct timer {
        timer(): start_t(std::chrono::steady_clock::now()) {}
        void stop() {
            stop_t = std::chrono::steady_clock::now();
        }
        std::chrono::steady_clock::duration elapsed() const {
            return stop_t - start_t;
        }
        friend std::ostream& operator<<(std::ostream& out, timer const& t) {
            return out << std::chrono::duration_cast<std::chrono::milliseconds>(t.elapsed()).count() << "ms";
        }
        std::chrono::steady_clock::time_point start_t, stop_t;
    };
    timer alloc, send, readable_domain, readable, copy, view, writeable;

    void alloc_begin(std::shared_ptr<pressio_domain> const& domain, pressio_dtype dtype, std::vector<size_t> const& dims) override {
        std::cout << "alloc_begin(" << prefix_or_name(*domain) << ") " << dtype << " {";
        for(size_t d: dims) {
            std::cout << d << ", ";
        }
        std::cout << '}' << std::endl;
        alloc = timer();
    };
    /**
     * called after memory is allocated from a domain
     */
    void alloc_end(std::shared_ptr<pressio_domain> const& domain, pressio_dtype dtype, std::vector<size_t> const& dims ) override {
        (void)dims;
        (void)dtype;
        alloc.stop();
        std::cout << "alloc_end(" << prefix_or_name(*domain) << ") " << alloc << std::endl;
    }
    /**
     * called before memory is viewed from a domain
     */
    void view_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::cout << "view_begin(" << prefix_or_name(*dst) << "<-" << prefix_or_name(src.domain()) << ')' << ' ' << src << std::endl;
        view = timer();
    }

    void view_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        view.stop();
        std::cout << "view_end(" << prefix_or_name(*dst) << "<-" << prefix_or_name(src.domain()) << ") " << view << std::endl;
    }
    void send_begin(pressio_data const& dst, pressio_data const& src) override {
        std::cout << "send_begin(" << prefix_or_name(*dst.domain()) << "<-" << prefix_or_name(src.domain()) << ") " << src << std::endl;
        send = timer();
    }
    void send_end(pressio_data const& dst, pressio_data const& src) override {
        send.stop();
        std::cout << "send_end(" << prefix_or_name(*dst.domain()) << "<-" << prefix_or_name(src.domain()) << ") " << send << std::endl;
    };
    void make_readable_begin(pressio_data const& dst, pressio_data const& src) override {
        std::cout << "readable_begin(" << prefix_or_name(*dst.domain()) << "<-" << prefix_or_name(src.domain()) << ") " << src << std::endl;
        readable = timer();
    }
    void make_readable_end(pressio_data const& dst, pressio_data const& src) override {
        readable.stop();
        std::cout << "readable_end(" << prefix_or_name(*dst.domain()) << "<-" << prefix_or_name(src.domain()) << ')' << readable << std::endl;

    }
    void make_readable_domain_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        readable_domain = timer();
        std::cout << "readable_domain_begin(" << prefix_or_name(*dst) << "<-" << prefix_or_name(src.domain()) << ") " << src << std::endl;
    }
    void make_readable_domain_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        readable_domain.stop();
        std::cout << "readable_domain_end(" << prefix_or_name(*dst) << "<-" << prefix_or_name(src.domain()) << ") " << readable_domain << std::endl;
    }
    void copy_to_begin(pressio_data const& dst, pressio_data const& src) override {
        copy = timer();
        std::cout << "copy_to_begin(" << prefix_or_name(*dst.domain()) << "<-" << prefix_or_name(src.domain()) << ") " << src << std::endl;
    }
    void copy_to_end(pressio_data const& dst, pressio_data const& src) override {
        copy.stop();
        std::cout << "copy_to_end(" << prefix_or_name(*dst.domain()) << "<-" << prefix_or_name(src.domain()) << ") " << copy << std::endl;
    }
    void make_writeable_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::cout << "writeable_begin(" << prefix_or_name(dst) << "<-" << prefix_or_name(src.domain()) << ") " << src << std::endl;
        writeable = timer();
    };
    void make_writeable_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        writeable.stop();
        std::cout << "writeable_end(" << prefix_or_name(dst) << "<-" << prefix_or_name(src.domain()) << ')' << writeable << std::endl;
    };

    domain_options get_metrics_results() override {
        domain_options opts;
        set(opts, get_name(), "print:alloc", alloc.elapsed().count());
        set(opts, get_name(), "print:view", view.elapsed().count());
        set(opts, get_name(), "print:send", send.elapsed().count());
        set(opts, get_name(), "print:copy", copy.elapsed().count());
        set(opts, get_name(), "print:make_readable", readable.elapsed().count());
        set(opts, get_name(), "print:make_writeable", writeable.elapsed().count());
        set(opts, get_name(), "print:make_readable_domain", readable_domain.elapsed().count());
        return opts;
    }

    std::unique_ptr<pressio_domain_manager_metrics_plugin> clone() const override {
        return std::make_unique<print_plugin>();
    }

    const char* prefix() const override {
        return "print";
    }

};

pressio_register X(domain_metrics_plugins(), "print", []{ return std::make_unique<print_plugin>();} );
}}}
