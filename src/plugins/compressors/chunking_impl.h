#ifndef LIBPRESSIO_CHUNKING_IMPL
#define LIBPRESSIO_CHUNKING_IMPL

#include <cstddef>
#include <vector>

struct pressio_data;
struct pressio_options;

namespace libpressio {
namespace compressors {
namespace chunking {

/**
 * preform the chunking as optimal-ally we know how to
 *
 * For 1d - 3d it dispatches to a handcoded loop; the handcoded loop can be
 * paralleled if OpenMP is availible
 *
 * support is availible For 4d+ it dispatches generic loop
 *
 * \param[in] data the dataset to preform chunking on
 * \param[in] block the dimensions to for the block size
 * \param[in] options options for how to chunk the data, mostly for performance
 * \returns the data in a single pressio_data buffer 
 */
pressio_data chunk_data(pressio_data const &data, std::vector<size_t> const &block, pressio_options const&);
  
/**
 * preform the chunking as optimal-ally we know how to
 *
 * For 1d - 3d it dispatches to a handcoded loop; the handcoded loop can be
 * paralleled if OpenMP is availible
 *
 * support is availible For 4d+ it dispatches generic loop
 *
 * \param[in] data the dataset to write the de-chunked memory into
 * \param[in] memory the dataset containing the chunked data
 * \param[in] block the dimensions to for the block size
 * \param[in] options options for how to chunk the data, mostly for performance
 * \returns the data in a single pressio_data buffer 
 */
void restore_data(pressio_data &data, pressio_data const& memory, std::vector<size_t> const &block, pressio_options const&);
  




} /* chunking */ 
} /* compressors */ 
} /* pressio */ 

#endif /* end of include guard: LIBPRESSIO_CHUNKING_IMPL */
