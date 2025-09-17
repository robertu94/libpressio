#include <libpressio_ext/cpp/domain_manager.h>

libpressio::domains::pressio_domain_manager& domain_manager() {
    static libpressio::domains::pressio_domain_manager mgr;
    return mgr;
}
/**
 * the registry for metrics plugins
 */
libpressio::pressio_registry<std::unique_ptr<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin>>& domain_metrics_plugins() {
    static libpressio::pressio_registry<std::unique_ptr<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin>> reg;
    return reg;
}

namespace libpressio { namespace domain_metrics { namespace noop_ns {
pressio_register registration(domain_metrics_plugins(), "noop", []{ return std::make_unique<libpressio::domains_metrics::pressio_domain_manager_metrics_plugin>();} );
}}}
