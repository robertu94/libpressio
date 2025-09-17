#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/domain_send.h>

namespace libpressio { namespace domains {
struct pressio_domain_send_host_to_device: public pressio_domain_send {
    void send(pressio_data& dst, pressio_data const& src) const override;
};
struct pressio_domain_send_device_to_host: public pressio_domain_send {
    void send(pressio_data& dst, pressio_data const& src) const override;
};
}}
