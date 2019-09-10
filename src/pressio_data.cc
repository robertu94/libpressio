#include <iostream>
#include <iterator>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstring>
#include <algorithm>
#include <functional>
#include <numeric>
#include "std_compat.h"
#include "pressio_data.h"
#include "multi_dimensional_iterator.h"
#include "libpressio_ext/cpp/data.h"


void pressio_data_libc_free_fn(void* data, void*) {
  free(data);
}

namespace {
  bool validate_select_args(std::vector<size_t> const& start,
    std::vector<size_t> const& stride,
    std::vector<size_t> const& count,
    std::vector<size_t> const& block,
    std::vector<size_t> const& dims)
  {
    if(dims.size() == 0)
      return false;

    if(start.size() != dims.size() ||
       stride.size() != dims.size() ||
       count.size() != dims.size() ||
       block.size() != dims.size()) {
      return false;
    }

    auto at_least_one = [](size_t i){ return i >= 1; };
    if(not (all_of(begin(count), end(count), at_least_one) &&
            all_of(begin(block), end(block), at_least_one) &&
            all_of(begin(stride), end(stride), at_least_one))) {
      return false;
    }

    std::vector<int> out_of_bounds(dims.size());
    for (unsigned long i = 0; i < dims.size(); ++i) {
      /*
       * start is the inital skip
       * block is the final skip
       * stride*(count-1) is the skip due to strided blocks
       * -1 is to not count the last block
       */
      out_of_bounds[i] = ((start[i] + block[i] + (count[i] - 1) * stride[i] - 1) > dims[i]);
    }

    auto is_true = [](auto v){ return v == true; };
    if(any_of(begin(out_of_bounds), end(out_of_bounds), is_true)) {
      return false;
    }

    return true;
  }

  struct copy_multi_dims_args {
    std::vector<size_t> const& global_dims;
    std::vector<size_t> const& stride;
    std::vector<size_t> const& count;
    std::vector<size_t> const& block;
    std::vector<size_t> const& start;
  };

  template <class Type>
  void copy_multi_dims(void* src, void* out, copy_multi_dims_args const& args) {
    Type* source = static_cast<Type*>(src);
    Type* dest = static_cast<Type*>(out);
    std::vector<size_t> const ones(args.global_dims.size(), 1);
    auto blocks = std::make_shared<multi_dimensional_range<Type>>(source,
        std::begin(args.global_dims),
        std::end(args.global_dims),
        std::begin(args.count),
        std::begin(args.stride),
        std::begin(args.start)
        );

    {
      auto block = std::begin(*blocks);
      auto block_end = std::end(*blocks);
      for(; block != block_end; ++block) {
        auto block_it = std::make_shared<multi_dimensional_range<Type>>(block,
            std::begin(args.global_dims),
            std::begin(args.block),
            std::begin(ones)
            );
        dest = std::copy(std::begin(*block_it), std::end(*block_it), dest);
        
      }
    }
  }
}


pressio_data pressio_data::select(std::vector<size_t> const& start,
    std::vector<size_t> const& stride,
    std::vector<size_t> const& count,
    std::vector<size_t> const& block) const {
  if(not validate_select_args(start, stride, count, block, dimensions())) {
    return pressio_data::empty(dtype(), dimensions());
  }

  //compute output dimensions
  std::vector<size_t> output_dims(count.size());
  transform(begin(block), end(block), begin(count), begin(output_dims), std::multiplies{});

  //allocate output buffer
  auto output = pressio_data::owning(this->dtype(), output_dims);

  copy_multi_dims_args args {
    dimensions(),
    stride,
    count,
    block,
    start
  };

  switch(this->dtype())
  {
  case pressio_double_dtype: 
    copy_multi_dims<double>(data(), output.data(), args);
    break;
  case pressio_float_dtype:
    copy_multi_dims<float>(data(), output.data(), args);
    break;
  case pressio_uint8_dtype:
    copy_multi_dims<uint8_t>(data(), output.data(), args);
    break;
  case pressio_uint16_dtype:
    copy_multi_dims<uint16_t>(data(), output.data(), args);
    break;
  case pressio_uint32_dtype:
    copy_multi_dims<uint32_t>(data(), output.data(), args);
    break;
  case pressio_uint64_dtype:
    copy_multi_dims<uint64_t>(data(), output.data(), args);
    break;
  case pressio_int8_dtype:
    copy_multi_dims<int8_t>(data(), output.data(), args);
    break;
  case pressio_int16_dtype:
    copy_multi_dims<int16_t>(data(), output.data(), args);
    break;
  case pressio_int32_dtype:
    copy_multi_dims<int32_t>(data(), output.data(), args);
    break;
  case pressio_int64_dtype:
    copy_multi_dims<int64_t>(data(), output.data(), args);
    break;
  case pressio_byte_dtype:
  default:
    copy_multi_dims<void*>(data(), output.data(), args);
    break;
  }


  return output;
}

extern "C" {

struct pressio_data* pressio_data_select(
    struct pressio_data const* data,
    const size_t* start,
    const size_t* stride,
    const size_t* count,
    const size_t* block
    ) {
  size_t const dims = data->num_dimensions();
  std::vector<size_t> ones(dims, 1);
  std::vector<size_t> zeros(dims, 0);
  if(start == nullptr) start = zeros.data();
  if(stride == nullptr) stride = ones.data();
  if(count == nullptr) count = ones.data();
  if(block == nullptr) block = ones.data();

  return new pressio_data(data->select(
        std::vector<size_t>(start,start+dims),
        std::vector<size_t>(stride,stride+dims),
        std::vector<size_t>(count,count+dims),
        std::vector<size_t>(block,block+dims)
        ));
}

struct pressio_data* pressio_data_new_move(const enum pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[], pressio_data_delete_fn deleter, void* metadata) {
  return new pressio_data(pressio_data::move(dtype, data, num_dimensions, dimensions, deleter, metadata));
}

struct pressio_data* pressio_data_new_copy(const enum pressio_dtype dtype, void* src, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::copy(dtype, src, num_dimensions, dimensions));
}

struct pressio_data* pressio_data_new_owning(const enum pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::owning(dtype, num_dimensions, dimensions));
}

struct pressio_data* pressio_data_new_nonowning(const enum pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::nonowning(dtype, data, num_dimensions, dimensions));
}

struct pressio_data* pressio_data_new_empty(const pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::empty(dtype, num_dimensions, dimensions));
}

void pressio_data_free(struct pressio_data* data) {
  delete data;
}

void* pressio_data_copy(struct pressio_data const* data, size_t* out_bytes) {
  if(out_bytes != nullptr) {
    *out_bytes = data->size_in_bytes();
  } 

  void* copy = malloc(data->size_in_bytes());
  memcpy(copy, data->data(), data->size_in_bytes());
  return copy;
}

void* pressio_data_ptr(struct pressio_data const* data, size_t* out_bytes) {
  if(out_bytes != nullptr) *out_bytes = data->size_in_bytes();
  return data->data();
}

pressio_dtype pressio_data_dtype(struct pressio_data const* data) {
  return data->dtype();
}

bool pressio_data_has_data(struct pressio_data const* data) {
  return data->data() != nullptr;
}

size_t pressio_data_num_dimensions(struct pressio_data const* data) {
  return data->num_dimensions();
}

size_t pressio_data_get_dimension(struct pressio_data const* data, size_t dimension) {
  return data->get_dimension(dimension);
}

size_t pressio_data_get_bytes(struct pressio_data const* data) {
  return data->size_in_bytes();
}

size_t pressio_data_num_elements(struct pressio_data const* data) {
  return data->num_elements();
}


}
