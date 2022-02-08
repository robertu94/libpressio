#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "pressio_data.h"
#include "multi_dimensional_iterator.h"
#include "libpressio_ext/cpp/data.h"
#include "std_compat/std_compat.h"


void pressio_data_libc_free_fn(void* data, void*) {
  free(data);
}

size_t data_size_in_elements(size_t dimensions, size_t const dims[]) {
  if(dimensions == 0) return 0;
  size_t totalsize = 1;
  for (size_t i = 0; i < dimensions; ++i) {
    totalsize *= dims[i];
  }
  return totalsize;
}
size_t data_size_in_bytes(pressio_dtype type, size_t const dimensions, size_t const dims[]) {
  return data_size_in_elements(dimensions, dims) * pressio_dtype_size(type);
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

    auto is_true = [](int v){ return v == true; };
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
    std::vector<size_t> const zeros(args.global_dims.size(), 0);
    std::vector<size_t> dest_global_dims(args.global_dims.size());
    std::transform(
        std::begin(args.block),
        std::end(args.block),
        std::begin(args.count),
        std::begin(dest_global_dims),
        compat::multiplies<>{}
        );

    auto src_blocks = std::make_shared<multi_dimensional_range<Type>>(source,
        std::begin(args.global_dims),
        std::end(args.global_dims),
        std::begin(args.count),
        std::begin(args.stride),
        std::begin(args.start)
        );
    auto dst_blocks = std::make_shared<multi_dimensional_range<Type>>(dest,
        std::begin(dest_global_dims),
        std::end(dest_global_dims),
        std::begin(args.count),
        std::begin(args.block),
        std::begin(zeros)
        );

    {
      auto src_block = std::begin(*src_blocks);
      auto src_block_end = std::end(*src_blocks);
      auto dst_block = std::begin(*dst_blocks);
      auto dst_block_end = std::end(*dst_blocks);
      for(; src_block != src_block_end; ++src_block, ++dst_block) {
        auto src_block_it = std::make_shared<multi_dimensional_range<Type>>(src_block,
            std::begin(args.block),
            std::begin(ones)
            );
        auto dst_block_it = std::make_shared<multi_dimensional_range<Type>>(dst_block,
            std::begin(args.block),
            std::begin(ones)
            );
        std::copy(std::begin(*src_block_it), std::end(*src_block_it), std::begin(*dst_block_it));
        
      }
    }
  }

  struct cast_fn {
    template <class T, class V>
    int operator()(T* src_begin, T* src_end, V* dst_begin, V*dst_end) {
      size_t num_elements = std::min(dst_end-dst_begin, src_end-src_begin);
      std::copy_n(src_begin, num_elements, dst_begin);
      return 0;
    }
};


  struct data_all_equal {

    template <class T, class V> 
    bool operator()(T const* lhs_begin, T const* lhs_end, V const* rhs_begin, V const*) {
      return std::equal(lhs_begin, lhs_end, rhs_begin);
    }
    
  };
}


pressio_data pressio_data::select(std::vector<size_t> const& start,
    std::vector<size_t> const& stride,
    std::vector<size_t> const& count,
    std::vector<size_t> const& block) const {
  std::vector<size_t> ones(dims.size(), 1);
  std::vector<size_t> zeros(dims.size(), 0);

  std::vector<size_t> const& real_start = start.empty() ? zeros : start;
  std::vector<size_t> const& real_stride = stride.empty() ? ones: stride;
  std::vector<size_t> const& real_count = count.empty() ? ones: count;
  std::vector<size_t> const& real_block = block.empty() ? ones: block;


  if(not validate_select_args(real_start, real_stride, real_count, real_block, dimensions())) {
    return pressio_data::empty(dtype(), dimensions());
  }

  //compute output dimensions
  std::vector<size_t> output_dims(real_count.size());
  transform(begin(real_block), end(real_block), begin(real_count), begin(output_dims), compat::multiplies<>{});

  //allocate output buffer
  auto output = pressio_data::owning(this->dtype(), output_dims);

  copy_multi_dims_args args {
    dimensions(),
    real_stride,
    real_count,
    real_block,
    real_start
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
  case pressio_bool_dtype:
    copy_multi_dims<bool>(data(), output.data(), args);
    break;
  case pressio_byte_dtype:
  default:
    copy_multi_dims<void*>(data(), output.data(), args);
    break;
  }


  return output;
}

namespace {
  struct transpose_impl {
    template <class RandomIt1, class RandomIt2>
    bool operator()(RandomIt1 r1_begin, RandomIt1, RandomIt2 r2_begin, RandomIt2) {
      std::vector<size_t> iter(dimensions.size(), 0);
      std::vector<size_t> strides(dimensions.size(), 0);
      size_t max_idx = std::accumulate(std::begin(dimensions), std::end(dimensions), 1, compat::multiplies<>{});
      compat::exclusive_scan(compat::rbegin(iter_max), compat::rend(iter_max), compat::rbegin(strides), 1, compat::multiplies<>{});
      for (size_t src_idx = 0; src_idx < max_idx; ++src_idx) {
        size_t dst_idx = compat::transform_reduce(std::begin(iter), std::end(iter), std::begin(strides), 0, compat::plus<>{}, compat::multiplies<>{});
        //clang-tidy treats this line as bugprone conversion since
        //it can cause unintended behavior in switch-case statements 
        //matching against character codes.  Since we aren't doing
        //that here, ignore this warning.
        r2_begin[src_idx] = r1_begin[dst_idx]; //NOLINT

        size_t idx = 0;
        bool updating = true;
        while(updating) {
          ++iter[idx];
          if(iter[idx] == iter_max[idx]) {
            iter[idx] = 0;
            if(++idx ==  iter_max.size()) {
              updating = false;
            } else {
              updating = true;
            }
          } else {
            updating = false;
          }
        }
      }
      
      return 0;
    }

    std::vector<size_t> const iter_max;
    std::vector<size_t> const dimensions;
  };
}

pressio_data pressio_data::transpose(std::vector<size_t> const& axis) const {
  auto ret = pressio_data::clone(*this);
  auto const iter_max = [&, this](){
    std::vector<size_t> pos(dims.size());
    if(axis.empty()) {
      std::copy(compat::rbegin(dims), compat::rend(dims), std::begin(pos));
    } else {
      for (size_t i = 0; i < dims.size(); ++i) {
        pos[i] = dims[axis[i]];
      }
    }
    return pos;
  }();
  ret.reshape(iter_max);
  pressio_data_for_each<int>(*this, ret, transpose_impl{iter_max, dimensions()});
  return ret;
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

struct pressio_data* pressio_data_transpose(
    struct pressio_data const* data,
    const size_t* axis
    ) {
  std::vector<size_t> axis_v;
  if(axis == nullptr) axis_v = {};
  else axis_v = std::vector<size_t>(axis, axis + data->num_dimensions());
  return new pressio_data(data->transpose(axis_v));
}

struct pressio_data* pressio_data_new_clone(const struct pressio_data* src) {
  return new pressio_data(pressio_data::clone(*src));
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

struct pressio_data* pressio_data_cast(const struct pressio_data* data, const enum pressio_dtype dtype) {
  return new pressio_data(data->cast(dtype));
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
  return data->has_data();
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

size_t pressio_data_get_capacity_in_bytes(struct pressio_data const* data) {
  return data->capacity_in_bytes();
}

int pressio_data_reshape(struct pressio_data* data,
    size_t const num_dimensions,
    size_t const dimensions[]
    ) {
  std::vector<size_t> new_dims(dimensions, dimensions+num_dimensions);
  return data->reshape(new_dims);
}

pressio_data pressio_data::cast(pressio_dtype const dtype) const {
    pressio_data data = pressio_data::owning(dtype, dimensions());
    pressio_data_for_each<int>(*this, data, cast_fn());
    return data;
}

bool pressio_data::operator==(pressio_data const& rhs) const {
  if(data_dtype != rhs.data_dtype) return false;
  if(dims != rhs.dims) return false;
  return pressio_data_for_each<bool>(*this, rhs, data_all_equal{});
}

}
