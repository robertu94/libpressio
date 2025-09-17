#include <memory>
#include <libpressio_ext/cpp/registry.h>
#include <libpressio_ext/cpp/domain_send.h>
namespace libpressio {
pressio_registry<std::unique_ptr<domains::pressio_domain_send>>& domain_send_plugins() {
    static pressio_registry<std::unique_ptr<domains::pressio_domain_send>> plugins;
    return plugins;
}
}
