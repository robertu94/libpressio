#include "cuda_domain_send.h"
namespace libpressio { namespace domains { namespace cuda_malloc_to_cudamalloc_ns {
pressio_register registration(domain_send_plugins(), "malloc>cudamalloc", []{return std::make_unique<pressio_domain_send_host_to_device>();});
}}}
