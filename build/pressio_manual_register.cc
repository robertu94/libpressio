
    #include <iostream>
    #include "libpressio_ext/cpp/registry.h"

    
    namespace libpressio {
	namespace compressors {
		namespace noop_ns {
			extern pressio_register registration;
		}
		namespace pressio_ns {
			extern pressio_register registration;
		}
	} /* namespace compressors*/
	namespace domains {
		namespace malloc_ns {
			extern pressio_register registration;
		}
		namespace nonowning_ns {
			extern pressio_register registration;
		}
	} /* namespace domains*/
	namespace domains_metrics {
		namespace print_ns {
			extern pressio_register registration;
		}
	} /* namespace domains_metrics*/
	namespace io {
		namespace by_extension_ns {
			extern pressio_register registration;
		}
		namespace noop_ns {
			extern pressio_register registration;
		}
		namespace posix_ns {
			extern pressio_register registration;
		}
	} /* namespace io*/
	namespace launch {
		namespace external_forkexec_ns {
			extern pressio_register registration;
		}
	} /* namespace launch*/
	namespace launch_metrics {
		namespace noop_ns {
			extern pressio_register registration;
		}
		namespace print_ns {
			extern pressio_register registration;
		}
	} /* namespace launch_metrics*/
	namespace metrics {
		namespace composite_ns {
			extern pressio_register registration;
		}
		namespace error_stat_ns {
			extern pressio_register registration;
		}
		namespace external_ns {
			extern pressio_register registration;
		}
		namespace noop_ns {
			extern pressio_register registration;
		}
	} /*namespace metrics*/
} /*namespace libpressio*/
extern "C" void pressio_register_all() {
libpressio::compressors::noop_ns::registration.ensure_registered();
libpressio::compressors::pressio_ns::registration.ensure_registered();
libpressio::domains::malloc_ns::registration.ensure_registered();
libpressio::domains::nonowning_ns::registration.ensure_registered();
libpressio::domains_metrics::print_ns::registration.ensure_registered();
libpressio::io::by_extension_ns::registration.ensure_registered();
libpressio::io::noop_ns::registration.ensure_registered();
libpressio::io::posix_ns::registration.ensure_registered();
libpressio::launch::external_forkexec_ns::registration.ensure_registered();
libpressio::launch_metrics::noop_ns::registration.ensure_registered();
libpressio::launch_metrics::print_ns::registration.ensure_registered();
libpressio::metrics::composite_ns::registration.ensure_registered();
libpressio::metrics::error_stat_ns::registration.ensure_registered();
libpressio::metrics::external_ns::registration.ensure_registered();
libpressio::metrics::noop_ns::registration.ensure_registered();

    }
    
