#include <libpressio_ext/cpp/domain_manager.h>

pressio_domain_manager& domain_manager() {
    static pressio_domain_manager mgr;
    return mgr;
}
/**
 * the registry for metrics plugins
 */
pressio_registry<std::unique_ptr<pressio_domain_manager_metrics_plugin>>& domain_metrics_plugins() {
    static pressio_registry<std::unique_ptr<pressio_domain_manager_metrics_plugin>> reg;
    return reg;
}

namespace libpressio { namespace domain_metrics { namespace noop {
pressio_register X(domain_metrics_plugins(), "noop", []{ return std::make_unique<pressio_domain_manager_metrics_plugin>();} );
}}}
