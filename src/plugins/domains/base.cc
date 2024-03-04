#include <libpressio_ext/cpp/domain.h>

pressio_registry<std::shared_ptr<pressio_domain>>& domain_plugins() {
    static pressio_registry<std::shared_ptr<pressio_domain>> registry;
    return registry;
}
