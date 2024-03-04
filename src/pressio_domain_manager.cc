#include <libpressio_ext/cpp/domain_manager.h>

static pressio_domain_manager mgr;

pressio_domain_manager& domain_manager() {
    return mgr;
}
