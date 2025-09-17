#include "cuda_domain_send.h"

namespace libpressio { namespace domains { namespace cuda_cudamallochost_to_cudamalloc_ns {
pressio_register registration(domain_send_plugins(), "cudamallochost>cudamalloc", []{return std::make_unique<pressio_domain_send_device_to_host>();});
}}}
