#include "cuda_domain_send.h"
namespace libpressio { namespace domains { namespace cuda_cudamalloc_to_cudamallochost_ns {
pressio_register registration(domain_send_plugins(), "cudamalloc>cudamallochost", []{return std::make_unique<pressio_domain_send_host_to_device>();});
}}}
