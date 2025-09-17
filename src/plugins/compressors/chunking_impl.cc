#include "chunking_impl.h"
#include <limits>
#include <vector>
#include <cstddef>
#include <numeric>
#include <std_compat/functional.h>
#include <std_compat/numeric.h>
#include <std_compat/iterator.h>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"

namespace libpressio {
namespace compressors {
namespace chunking {
namespace detail {

struct copy_from_blocks {
  template <class Data, class Block>
  static void in_bounds(Data& d, Block& b) noexcept {
    b = d;
  }

  template <class Data>
  static void out_of_bounds(Data&) noexcept {
  }
};
struct copy_to_blocks {
  template <class Data, class Block>
  static void in_bounds(Data& d, Block& b) noexcept {
    d = b;
  }

  template <class Data>
  static void out_of_bounds(Data& d) noexcept {
    d = 0;
  }
};

size_t working_memory_size(pressio_data const& data, std::vector<size_t> const& block) {
  auto dims = data.dimensions();
  auto dtype = data.dtype();
  std::vector<size_t> max_idx(dims.size());
  for (size_t i = 0; i < max_idx.size(); ++i) {
    if(dims[i] % block[i] == 0) {
      max_idx[i] = (dims[i]/block[i]);
    } else {
      max_idx[i] = (dims[i]/block[i]) + 1;
    }
  }
  size_t n_blocks = std::accumulate(std::begin(max_idx), std::end(max_idx), size_t{1}, compat::multiplies<>{});
  size_t block_size = std::accumulate(std::begin(block), std::end(block), size_t{1}, compat::multiplies<>{});
  return n_blocks * pressio_dtype_size(dtype) * block_size;
}

template <class CopyPolicy, class PressioData>
struct dispatch_1d {
  template <class T>
  int operator()(T* data_begin, T* ) {
    std::vector<size_t> max_idx(dims.size());
    std::vector<size_t> strides(dims.size());
    compat::exclusive_scan(compat::cbegin(dims), compat::cend(dims), std::begin(strides), size_t{1}, compat::multiplies<>{});
    size_t elements_in_block = 1;
    for (size_t i = 0; i < max_idx.size(); ++i) {
      if(dims[i] % block[i] == 0) {
        max_idx[i] = (dims[i]/block[i]);
      } else {
        max_idx[i] = (dims[i]/block[i]) + 1;
      }
      elements_in_block *= block[i];
    }
    std::vector<size_t> block_stide(block.size());
    compat::exclusive_scan(compat::cbegin(max_idx), compat::cend(max_idx), std::begin(block_stide), size_t{1}, compat::multiplies<>{});
#ifdef  _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for (size_t block_i = 0; block_i < max_idx[0]; block_i++) {
      size_t base_offset = block_i * block[0] * strides[0];
      size_t output_idx = block_i * block_stide[0];
      T* elements = static_cast<T*>(memory.data()) + (output_idx * elements_in_block );

      //could this block go out of bounds ever?
      if(block[0] * block_i + block[0] < dims[0]) {
        //we couldn't go out of bounds
        for (size_t element_i = 0; element_i < block[0]; ++element_i) {
          size_t offset = base_offset + element_i * strides[0];
          CopyPolicy::in_bounds(*elements, data_begin[offset]);
          ++elements;
        }
      } else {
        //we could go out of bounds
        for (size_t element_i = 0; element_i < block[0]; ++element_i) {
          if(element_i + block_i * block[0]  < dims[0]) {
            size_t offset = base_offset + element_i * strides[0];
            CopyPolicy::in_bounds(*elements, data_begin[offset]);
          } else {
            CopyPolicy::out_of_bounds(*elements);
          }
          ++elements;
        }
      }
    }

    return 0;
  }
  uint64_t nthreads;
  std::vector<size_t> const& dims;
  std::vector<size_t> const& block;
  PressioData& memory;
};

template <class CopyPolicy, class PressioData>
struct dispatch_2d {
  template <class T>
  int operator()(T* data_begin, T* ) {
  std::vector<size_t> max_idx(dims.size());
  std::vector<size_t> strides(dims.size());
  compat::exclusive_scan(compat::cbegin(dims), compat::cend(dims), std::begin(strides), size_t{1}, compat::multiplies<>{});
  size_t elements_in_block = 1;
  for (size_t i = 0; i < max_idx.size(); ++i) {
    if(dims[i] % block[i] == 0) {
      max_idx[i] = (dims[i]/block[i]);
    } else {
      max_idx[i] = (dims[i]/block[i]) + 1;
    }
    elements_in_block *= block[i];
  }
  std::vector<size_t> block_stide(block.size());
  compat::exclusive_scan(compat::cbegin(max_idx), compat::cend(max_idx), std::begin(block_stide), size_t{1}, compat::multiplies<>{});
  
#ifdef  _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
  for (size_t block_j = 0; block_j < max_idx[1]; block_j++) {
  for (size_t block_i = 0; block_i < max_idx[0]; block_i++) {
    size_t base_offset = block_i * block[0] * strides[0] + block_j * block[1] * strides[1];
    size_t output_idx = block_i * block_stide[0] + block_j * block_stide[1];
    T* elements = static_cast<T*>(memory.data()) + (output_idx * elements_in_block );

    //could this block go out of bounds ever?
    if(block[0] * block_i + block[0] < dims[0] &&
      block[1] * block_j + block[1] < dims[1]
        ) {
      //we couldn't go out of bounds
      for (size_t element_j = 0; element_j < block[1]; ++element_j) {
      for (size_t element_i = 0; element_i < block[0]; ++element_i) {
        size_t offset = base_offset + element_i * strides[0] + element_j * strides[1];
          CopyPolicy::in_bounds(*elements, data_begin[offset]);
          ++elements;
      }}
    } else {
      //we could go out of bounds
      for (size_t element_j = 0; element_j < block[1]; ++element_j) {
      for (size_t element_i = 0; element_i < block[0]; ++element_i) {
        if(element_i + block_i * block[0]  < dims[0] && 
           element_j + block_j * block[1] < dims[1]
            ) {
          size_t offset = base_offset + element_i * strides[0] + element_j * strides[1];
          CopyPolicy::in_bounds(*elements, data_begin[offset]);
        } else {
          CopyPolicy::out_of_bounds(*elements);
        }
        ++elements;
      }
    }}
  }}

    return 0;
  }
  uint64_t nthreads;
  std::vector<size_t> const& dims;
  std::vector<size_t> const& block;
  PressioData& memory;
};


template <class CopyPolicy, class PressioData>
struct dispatch_3d {
  template <class T>
  int operator()(T* data_begin, T* ) {
    std::vector<size_t> max_idx(dims.size());
    std::vector<size_t> strides(dims.size());
    compat::exclusive_scan(compat::cbegin(dims), compat::cend(dims), std::begin(strides), 1, compat::multiplies<>{});
    size_t elements_in_block = 1;
    for (size_t i = 0; i < max_idx.size(); ++i) {
      if(dims[i] % block[i] == 0) {
        max_idx[i] = (dims[i]/block[i]);
      } else {
        max_idx[i] = (dims[i]/block[i]) + 1;
      }
      elements_in_block *= block[i];
    }
    std::vector<size_t> block_stide(block.size());
    compat::exclusive_scan(compat::cbegin(max_idx), compat::cend(max_idx), std::begin(block_stide), size_t{1}, compat::multiplies<>{});
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for (size_t block_k = 0; block_k < max_idx[2]; block_k++) {
    for (size_t block_j = 0; block_j < max_idx[1]; block_j++) {
    for (size_t block_i = 0; block_i < max_idx[0]; block_i++) {
      size_t base_offset = block_i * block[0] * strides[0] + block_j * strides[1] * block[1] + block_k * strides[2] * block[2];
      size_t output_idx = block_i * block_stide[0] + block_j * block_stide[1] + block_k * block_stide[2];
      T* elements = static_cast<T*>(memory.data()) + (output_idx * elements_in_block );

      if(block[0] * block_i + block[0] < dims[0] &&
        block[1] * block_j + block[1] < dims[1] &&
        block[2] * block_k + block[2] < dims[2]
          ) {
        //we couldn't go out of bounds
        for (size_t element_k = 0; element_k < block[2]; ++element_k) {
        for (size_t element_j = 0; element_j < block[1]; ++element_j) {
        for (size_t element_i = 0; element_i < block[0]; ++element_i) {
          size_t offset = base_offset + element_i * strides[0] + element_j * strides[1] + element_k * strides[2];
          CopyPolicy::in_bounds(*elements, data_begin[offset]);
          ++elements;
        }}}
      } else {
        //we could go out of bounds
        for (size_t element_k = 0; element_k < block[2]; ++element_k) {
        for (size_t element_j = 0; element_j < block[1]; ++element_j) {
        for (size_t element_i = 0; element_i < block[0]; ++element_i) {
          if(element_i + block_i * block[0]  < dims[0] && 
             element_j + block_j * block[1] < dims[1] &&
             element_k + block_k * block[2] < dims[2]
              ) {
            size_t offset = base_offset + element_i * strides[0] + element_j * strides[1] + element_k * strides[2];
            CopyPolicy::in_bounds(*elements, data_begin[offset]);
          } else {
            CopyPolicy::out_of_bounds(*elements);
          }
        ++elements;
        }}}
      }
    }}}

    return 0;
  }
  uint64_t nthreads;
  std::vector<size_t> const& dims;
  std::vector<size_t> const& block;
  PressioData& memory;
};

template <class CopyPolicy, class PressioData>
struct dispatch_generic {
  template <template <typename...> class vector_type, class IndexStride>
  static IndexStride incr_counter(vector_type<IndexStride>& counter, vector_type<IndexStride> const& max) noexcept {

    IndexStride i = 0;
    while(i < counter.size() && counter[i] == (max[i] - 1)) {
      counter[i] = 0;
      ++i;
    }
    if (i < counter.size()) {
      counter[i] += 1;
    }
    return i+1;
  }


  template <template <class...> class vector_type, class IndexStride, class IndexBlock, class IndexDims>
  static bool in_bounds(
      vector_type<IndexStride> const& start,
      vector_type<IndexBlock>  const& block_idx,
      vector_type<IndexDims>  const& dims
      ) noexcept {
    for (IndexStride i = 0; i < start.size(); ++i) {
      if (start[i] + block_idx[i] >= dims[i]){
        return false;
      }
    }

    return true;
  }


  template <class RandomIt, template <typename...> class vector_type, class elements_storage, class IndexStride, class IndexDims, class IndexBlock>
  static void
  prepare_block(RandomIt begin_it, vector_type<IndexDims> const& start,  vector_type<IndexDims> const& dims, vector_type<IndexBlock> const& block, vector_type<IndexStride> const& strides, vector_type<IndexBlock>& block_idx, 
      elements_storage elements
      ) noexcept {
    std::fill(begin(block_idx), end(block_idx), 0);

    bool done = false;
    while(not done) {
      if(in_bounds(start, block_idx, dims)) {
        IndexStride offset = 0;
        for (IndexStride i = 0; i < start.size(); ++i) {
           offset += (block_idx[i] + start[i]) * strides[i];
        }
        RandomIt it = std::next(begin_it, offset);
        CopyPolicy::in_bounds(*elements, *it);
      } else {
        CopyPolicy::out_of_bounds(*elements);
      }
      elements++;
      IndexStride modified = incr_counter(block_idx, block);
      if (modified == start.size() + 1) {
        done = true;
      } 
    }
  }

  template <class RandomIt, template <typename...> class vector_type, class elements_storage, class IndexStride, class IndexDims, class IndexBlock>
  static void
  prepare_block_in_bounds(RandomIt begin_it, vector_type<IndexDims> const& start,  vector_type<IndexDims> const& /*dims*/ , vector_type<IndexBlock> const& block, vector_type<IndexStride> const& strides, vector_type<IndexBlock>& block_idx, 
      elements_storage elements
      ) noexcept {
    std::fill(begin(block_idx), end(block_idx), 0);

    bool done = false;
    while(not done) {
      IndexStride offset = 0;
      for (IndexStride i = 0; i < start.size(); ++i) {
         offset += (block_idx[i] + start[i]) * strides[i];
      }
      RandomIt it = std::next(begin_it, offset);
      CopyPolicy::in_bounds(*elements, *it);
      ++elements;
      IndexStride modified = incr_counter(block_idx, block);
      if (modified == start.size() + 1) {
        done = true;
      } 
    }
  }

  template <class T, class IndexStride, class IndexDims, class IndexBlock>
  int work(T* data_begin, std::vector<IndexStride> const& stride, std::vector<IndexDims> const& dims, std::vector<IndexBlock> const& block) {
    std::vector<IndexDims> block_idx(dims.size());
    std::vector<IndexDims> max_idx(dims.size());
    std::vector<IndexDims> start(dims.size());
    std::vector<IndexBlock> inner_block_idx(start.size());
    size_t items_in_block = 1;
    for (IndexStride i = 0; i < max_idx.size(); ++i) {
      if(dims[i] % block[i] == 0) {
        max_idx[i] = (dims[i]/block[i]);
      } else {
        max_idx[i] = (dims[i]/block[i]) + 1;
      }
      items_in_block *= block[i];
    }
    bool done = false;

    T* elements = static_cast<T*>(memory.data());
    while(not done) {
      for (IndexStride i = 0; i < start.size(); ++i) {
        start[i] = block[i]*block_idx[i];
      }


      for (IndexStride i = 0; i < inner_block_idx.size(); ++i) {
        inner_block_idx[i] = block[i]-1;
      }
      if(in_bounds(start, inner_block_idx, dims)) {
        prepare_block_in_bounds(data_begin, start, dims, block, stride, inner_block_idx, elements);
      } else {
        prepare_block(data_begin, start, dims, block, stride, inner_block_idx, elements);
      }

      IndexStride modified = incr_counter(block_idx, max_idx);
      if(modified == start.size() + 1) {
        done = true;
      }
      elements += items_in_block;
    }

    return 0;
  }

  template <class T, class IndexStride, class IndexDims>
  int dispatch_block(T* data_begin, std::vector<IndexStride> const& stride, std::vector<IndexDims> const& dim, size_t max_block) {
    if(max_block <= std::numeric_limits<uint8_t>::max()) {
      std::vector<uint8_t> block(block_size_t.begin(), block_size_t.end());
      return work(data_begin, stride, dim, block);
    } else if(max_block <= std::numeric_limits<uint16_t>::max()) {
      std::vector<uint16_t> block(block_size_t.begin(), block_size_t.end());
      return work(data_begin, stride, dim, block);
    } else if(max_block <= std::numeric_limits<uint32_t>::max()) {
      std::vector<uint32_t> block(block_size_t.begin(), block_size_t.end());
      return work(data_begin, stride, dim, block);
    } else {
      std::vector<uint64_t> block(block_size_t.begin(), block_size_t.end());
      return work(data_begin, stride, dim, block);
    }
  }

  template <class T, class IndexStride>
  int dispatch_dims(T* data_begin, std::vector<IndexStride> const& stride, size_t max_dim, size_t max_block) {
    if(max_dim <= std::numeric_limits<uint8_t>::max()) {
      std::vector<uint8_t> dims(dims_size_t.begin(), dims_size_t.end());
      return dispatch_block(data_begin, stride, dims, max_block);
    } else if(max_dim <= std::numeric_limits<uint16_t>::max()) {
      std::vector<uint16_t> dims(dims_size_t.begin(), dims_size_t.end());
      return dispatch_block(data_begin, stride, dims, max_block);
    } else if(max_dim <= std::numeric_limits<uint32_t>::max()) {
      std::vector<uint32_t> dims(dims_size_t.begin(), dims_size_t.end());
      return dispatch_block(data_begin, stride, dims, max_block);
    } else {
      std::vector<uint64_t> dims(dims_size_t.begin(), dims_size_t.end());
      return dispatch_block(data_begin, stride, dims, max_block);
    }
  }

  template <class T>
  int operator()(T * data_begin, T * ) {
    size_t max_block_size = *std::max_element(std::begin(block_size_t), std::end(block_size_t));
    size_t max_dim_size = *std::max_element(std::begin(dims_size_t), std::end(dims_size_t));
    std::vector<size_t> strides(dims_size_t.size());
    compat::exclusive_scan(std::begin(dims_size_t), std::end(dims_size_t), std::begin(strides), size_t{1}, compat::multiplies<>{});
    if(std::is_sorted(std::begin(strides), std::end(strides))) {
      throw std::runtime_error("strides are too large");
    }
    size_t max_stride_size = *std::max_element(std::begin(strides), std::end(strides));

    if(max_stride_size <= std::numeric_limits<uint8_t>::max()) {
      std::vector<uint8_t> stride(strides.begin(), strides.end());
      return dispatch_dims(data_begin, stride, max_dim_size, max_block_size);
    } else if(max_stride_size <= std::numeric_limits<uint16_t>::max()) {
      std::vector<uint16_t> stride(strides.begin(), strides.end());
      return dispatch_dims(data_begin, stride, max_dim_size, max_block_size);
    } else if(max_stride_size <= std::numeric_limits<uint32_t>::max()) {
      std::vector<uint32_t> stride(strides.begin(), strides.end());
      return dispatch_dims(data_begin, stride, max_dim_size, max_block_size);
    } else {
      std::vector<uint64_t> stride(strides.begin(), strides.end());
      return dispatch_dims(data_begin, stride, max_dim_size, max_block_size);
    }
  
    return 0;
  }
  std::vector<size_t> const& dims_size_t;
  std::vector<size_t> const& block_size_t;
  PressioData& memory;
};

}

pressio_data chunk_data(pressio_data const& data, std::vector<size_t> const& block, pressio_options const& options) {
  uint64_t nthreads = 1;
  options.get("nthreads", &nthreads);

  pressio_data memory(pressio_data::owning(pressio_byte_dtype, {detail::working_memory_size(data, block)}));

  switch (data.num_dimensions()) {
    case 1:
      pressio_data_for_each<int>(data, detail::dispatch_1d<detail::copy_to_blocks, pressio_data>{nthreads, data.dimensions(), block, memory});
      break;
    case 2:
      pressio_data_for_each<int>(data, detail::dispatch_2d<detail::copy_to_blocks, pressio_data>{nthreads, data.dimensions(), block, memory});
      break;
    case 3:
      pressio_data_for_each<int>(data, detail::dispatch_3d<detail::copy_to_blocks, pressio_data>{nthreads, data.dimensions(), block, memory});
      break;
    default:
      pressio_data_for_each<int>(data, detail::dispatch_generic<detail::copy_to_blocks, pressio_data>{data.dimensions(), block, memory});
      break;
  }

  return memory;
}

void restore_data(
    pressio_data& data,
    pressio_data const& memory,
    std::vector<size_t> const& block,
    pressio_options const& options
    ) {
  uint64_t nthreads = 1;
  options.get("nthreads", &nthreads);


  switch (data.num_dimensions()) {
    case 1:
      pressio_data_for_each<int>(data, detail::dispatch_1d<detail::copy_from_blocks, const pressio_data>{nthreads, data.dimensions(), block, memory});
      break;
    case 2:
      pressio_data_for_each<int>(data, detail::dispatch_2d<detail::copy_from_blocks, const pressio_data>{nthreads, data.dimensions(), block, memory});
      break;
    case 3:
      pressio_data_for_each<int>(data, detail::dispatch_3d<detail::copy_from_blocks, const pressio_data>{nthreads, data.dimensions(), block, memory});
      break;
    default:
      pressio_data_for_each<int>(data, detail::dispatch_generic<detail::copy_from_blocks, const pressio_data>{data.dimensions(), block, memory});
      break;
  }

}

} /* chunking */ 
} /* pressio */ 
}

