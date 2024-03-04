#ifndef LIBPRESSIO_DOMAIN_SEND_H
#define LIBPRESSIO_DOMAIN_SEND_H

#include <memory>
#include <libpressio_ext/cpp/registry.h>
struct pressio_data;

struct pressio_domain_send {
    pressio_domain_send()=default;
    virtual ~pressio_domain_send()=default;

    virtual void send(pressio_data& dst, pressio_data const& src) const = 0;
};

pressio_registry<std::unique_ptr<pressio_domain_send>>& domain_send_plugins();

#endif /* end of include guard: LIBPRESSIO_DOMAIN_SEND_H */
