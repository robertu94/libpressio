#ifndef LIBPRESSIO_DOMAIN_SEND_H
#define LIBPRESSIO_DOMAIN_SEND_H

#include <memory>
#include <libpressio_ext/cpp/registry.h>

struct pressio_data;

namespace libpressio { namespace domains {

    /**
     * a plugin that understands how to transfer data from one memory domain (e.g. malloc, cudamalloc) to another
     */
struct pressio_domain_send {
    /**
     * create an instance of a send plugin
     */
    pressio_domain_send()=default;
    /**
     * destroy an instance of a send plugin
     */
    virtual ~pressio_domain_send()=default;

    /**
     * copy data from src to dst.  Throws an exception on error
     * \param[in] src the source data buffer
     * \param[in] dst the destination data buffer
     */
    virtual void send(pressio_data& dst, pressio_data const& src) const = 0;
};
}
/**
 * registry for all send plugins
 */
pressio_registry<std::unique_ptr<domains::pressio_domain_send>>& domain_send_plugins();
}

#endif /* end of include guard: LIBPRESSIO_DOMAIN_SEND_H */
