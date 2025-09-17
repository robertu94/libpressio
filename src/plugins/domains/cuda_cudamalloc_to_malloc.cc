#include "cuda_domain_send.h"
namespace libpressio {namespace domains { namespace cuda_cudamalloc_to_malloc_ns {
pressio_register registration(domain_send_plugins(), "cudamalloc>malloc", []{return std::make_unique<pressio_domain_send_device_to_host>();});
}}}
