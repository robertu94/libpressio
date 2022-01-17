#include "std_compat/functional.h"
#include "std_compat/memory.h"
#include "std_compat/optional.h"
#include <numeric>
#include <chrono>
#include "pressio_metrics.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/metrics.h"
#include <mgard/MGARDConfig.hpp>
#include <mgard/compress.hpp>
#include <mgard/TensorQuantityOfInterest.hpp>

namespace libpressio { namespace mgard10_ns {

namespace qoi {
struct caller {
  template <class T, class V, size_t N>
  T operator()(mgard::TensorMeshHierarchy<N,T> const& h, V const* data) const {
    auto dims = h.shapes[h.L];
    if (supports_dtype(pressio_dtype_from_type<V>())) {
      return call(compat::span<size_t>(dims.data(), dims.size()), static_cast<const void*>(data));
    } else {
      throw std::invalid_argument("invalid type");
    }
  }

  virtual bool supports_dtype(pressio_dtype dtype) const =0;
  virtual double call(compat::span<size_t>, const void* data) const = 0;
  virtual std::unique_ptr<caller> clone() const = 0;
  virtual void set_dtype(pressio_dtype dtype) = 0;
};

struct caller_ptr {
  caller_ptr(): ptr() {}
  caller_ptr(caller_ptr const& lhs): ptr(lhs.ptr->clone()) {}
  caller_ptr(caller_ptr && lhs) noexcept: ptr(std::move(lhs.ptr)) {}
  caller_ptr& operator=(caller_ptr const& lhs) {
    if(this == &lhs) return *this;
    ptr = lhs.ptr->clone();
    return *this;
  }
  caller_ptr& operator=(caller_ptr && lhs) noexcept {
    if(this == &lhs) return *this;
    ptr = std::move(lhs.ptr);
    return *this;
  }
  caller_ptr(std::unique_ptr<caller> && caller): ptr(std::move(caller)) {}

  caller* operator->() {
    return ptr.operator->();
  }
  caller& operator*() {
    return ptr.operator*();
  }
  operator bool() {
    return ptr.operator bool();
  }

  std::unique_ptr<caller> ptr;
};

template <class T>
struct call_cfunc final: public caller {
  call_cfunc(T (*func)(int,int,int, T const*)) : func(func) {}

  double call(compat::span<size_t> dims, void const* d) const override {
    const T* data = static_cast<const T*>(d);
    return func(
        dims.size() > 0 ? dims[0] : 1,
        dims.size() > 1 ? dims[1] : 1,
        dims.size() > 2 ? dims[2] : 1,
        data
        );
  }
  bool supports_dtype(pressio_dtype dtype) const override {
    return dtype == pressio_dtype_from_type<T>();
  }
  void set_dtype(pressio_dtype) override { }

  T (*func)(int,int,int, const T*);
  std::unique_ptr<caller> clone() const override {
    return compat::make_unique<call_cfunc<T>>(func);
  }
};

template <class T>
struct call_cfuncv final : public caller {
  call_cfuncv(T (*func)(int,int,int, T const*, void*), void* info) : func(func), info(info) {}
  double call (compat::span<size_t> dims, const void* data) const override {
    T const* d = static_cast<T const*>(data);
    return func(
        dims.size() > 0 ? dims[0] : 1,
        dims.size() > 1 ? dims[1] : 1,
        dims.size() > 2 ? dims[2] : 1,
        d,
        info
        );
  }
  bool supports_dtype(pressio_dtype dtype) const override {
    return dtype == pressio_dtype_from_type<T>();
  }

  std::unique_ptr<caller> clone() const override {
    return compat::make_unique<call_cfuncv<T>>(func, info);
  }
  void set_dtype(pressio_dtype) override { }

  T (*func)(int,int,int, const T*, void*);
  void* info;
};

struct call_pressio_metric final : public caller {
  call_pressio_metric(std::string const& output_metric, pressio_metrics&& metric, pressio_dtype last):
    output_metric(output_metric), metric(metric), last(last) 
  {
  }

  double call (compat::span<size_t> dims, const void* data) const override {
    pressio_data lp_data(pressio_data::nonowning(
          last,
          const_cast<void*>(data),
          std::vector<size_t>(dims.begin(), dims.end())
          ));
    pressio_metrics metric = this->metric->clone();
    pressio_options* results = pressio_metrics_evaluate(
        &metric,
        nullptr,
        nullptr,
        &lp_data
        );
    double result = 0.0;
    results->cast(output_metric, &result, pressio_conversion_explicit);
    pressio_options_free(results);
    return result;
  }

  bool supports_dtype(pressio_dtype) const override {
    return true;
  }
  void set_dtype(pressio_dtype dtype) override {
    last = dtype;
  }
  std::unique_ptr<caller> clone() const override {
    return compat::make_unique<call_pressio_metric>(output_metric, pressio_metrics(metric->clone()), last);
  }
	
  std::string output_metric;
  pressio_metrics metric;
  pressio_dtype last;
};
}

namespace cpu {

struct compress_operator {
  template <class T>
  pressio_data operator()(T* begin, T*) {
    switch(shape.size()) {
      case 1:
        return compress_impl<T, 1>(begin);
      case 2:
        return compress_impl<T, 2>(begin);
      case 3:
        return compress_impl<T, 3>(begin);
      default:
        throw std::runtime_error("unsupported number of dimensions");
    }
  }

  template <class T, size_t N>
  pressio_data compress_impl(T* begin) {
    compat::optional<mgard::TensorMeshHierarchy<N, T>> hierarchy;
    std::array<size_t, N> mgard_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      mgard_shape[i] = shape[i];
    }
    if(coordinates) {
      std::array<std::vector<T>, N> mgard_coordinates;
      const size_t input_size = std::accumulate(shape.begin(), shape.end(), size_t{1}, compat::multiplies{});
      T* coordinates_ptr = static_cast<T*>(coordinates->data());
      for (size_t i = 0; i < N; ++i) {
        mgard_coordinates[i] = std::vector<T>(coordinates_ptr+(i*input_size), coordinates_ptr+((i+1)* input_size));
      }
      hierarchy = mgard::TensorMeshHierarchy<N, T> (mgard_shape, mgard_coordinates);
    } else {
      hierarchy = mgard::TensorMeshHierarchy<N, T> (mgard_shape);
    }
    if(recompute_qoi && caller_func) {
      auto begin = std::chrono::steady_clock::now();
      caller_func->set_dtype(pressio_dtype_from_type<T>());
      mgard::TensorQuantityOfInterest<N,T> qoi(*hierarchy, *caller_func);
      norm_of_qoi = qoi.norm(s);
      auto end = std::chrono::steady_clock::now();
      norm_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
    }
    mgard::CompressedDataset<N, T> const compressed = mgard::compress(*hierarchy, begin, static_cast<T>(s), static_cast<T>(tolerance * norm_of_qoi.value_or(1)));
    return pressio_data::copy(pressio_byte_dtype, compressed.data(), {compressed.size()});
  }

  compat::optional<pressio_data> const& coordinates;
  std::vector<size_t> const& shape;
  double s, tolerance;
  compat::optional<double>& norm_of_qoi;
  qoi::caller_ptr& caller_func;
  bool recompute_qoi;
  compat::optional<uint64_t>& norm_time;
};

struct decompress_operator {
  template <class OutputType>
  pressio_data operator()(const unsigned char* begin, const unsigned char* end) {
    auto decompressed = mgard::decompress(begin, std::distance(begin, end));
    return pressio_data::copy(pressio_dtype_from_type<OutputType>(), decompressed.get(), shape);
  }
  std::vector<size_t> const& shape;
};
}

namespace gpu {

struct compress_operator {
  template <class T>
  pressio_data operator()(T* begin, T*) {
    switch(shape.size()) {
      case 1:
        return compress_impl<T, 1>(begin);
      case 2:
        return compress_impl<T, 2>(begin);
      case 3:
        return compress_impl<T, 3>(begin);
      default:
        throw std::runtime_error("unsupported number of dimensions");
    }
  }

  template <class T, size_t N>
  pressio_data compress_impl(T* begin) {
    compat::optional<mgard::TensorMeshHierarchy<N, T>> hierarchy;
    std::vector<mgard_cuda::SIZE> gpu_shape(N);
    std::array<size_t, N> mgard_shape;
    size_t num_elements = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      mgard_shape[i] = shape[i];
      gpu_shape[i] = shape[i];
      num_elements *= shape[i];
    }
    if(recompute_qoi && caller_func) {
      auto begin = std::chrono::steady_clock::now();
      caller_func->set_dtype(pressio_dtype_from_type<T>());
      mgard::TensorQuantityOfInterest<N,T> qoi(*hierarchy, *caller_func);
      norm_of_qoi = qoi.norm(s);
      auto end = std::chrono::steady_clock::now();
      norm_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
    }

    T* in_array_cpu = nullptr;
    mgard_cuda::cudaMallocHostHelper( (void**)&in_array_cpu, sizeof(T)*num_elements);
    std::copy(begin, begin + num_elements, in_array_cpu);

    mgard_cuda::error_bound_type error_bound_type = mgard_cuda::error_bound_type::ABS;
    mgard_cuda::Handle<N, T> handle(gpu_shape, config);
    mgard_cuda::Array<N, T> in_array(gpu_shape);
    in_array.loadData(in_array_cpu);
    mgard_cuda::Array<1, unsigned char> compressed = mgard_cuda::compress(
        handle, in_array, error_bound_type, static_cast<T>(tolerance), static_cast<T>(s)
        );
    mgard_cuda::cudaFreeHostHelper(in_array_cpu);
    return pressio_data::copy(pressio_byte_dtype, compressed.getDataHost(), {compressed.getShape()[0]});

  }

  compat::optional<pressio_data> const& coordinates;
  std::vector<size_t> const& shape;
  double s, tolerance;
  compat::optional<double>& norm_of_qoi;
  qoi::caller_ptr& caller_func;
  bool recompute_qoi;
  compat::optional<uint64_t>& norm_time;
  mgard_cuda::Config& config;
};

struct decompress_operator {
  template <class OutputType>
  pressio_data operator()(const unsigned char* begin, const unsigned char* end) {
    switch(shape.size()) {
      case 1:
        return decompress_impl<OutputType, 1>(begin, end);
      case 2:
        return decompress_impl<OutputType, 2>(begin, end);
      case 3:
        return decompress_impl<OutputType, 3>(begin, end);
      case 4:
        return decompress_impl<OutputType, 4>(begin, end);
      case 5:
        return decompress_impl<OutputType, 5>(begin, end);
      default:
        throw std::runtime_error("unsupported number of dimensions");
    }
  }

  template <class T, size_t N>
  pressio_data decompress_impl(const unsigned char* begin, const unsigned char* end) {
    std::vector<mgard_cuda::SIZE> gpu_out_shape(N);
    std::vector<mgard_cuda::SIZE> gpu_in_shape(1);
    gpu_in_shape[0] = std::distance(begin, end);
    size_t num_elements = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      gpu_out_shape[i] = shape[i];
      num_elements *= shape[i];
    }
    unsigned char* in_array_cpu = nullptr;
    mgard_cuda::cudaMallocHostHelper((void**)&in_array_cpu, sizeof(unsigned char)*std::distance(begin,end));
    std::copy(begin, end, in_array_cpu);

    mgard_cuda::Handle<N, T> handle(gpu_out_shape, config);
    mgard_cuda::Array<1, unsigned char> in_array(gpu_in_shape);
    in_array.loadData(in_array_cpu);
    mgard_cuda::Array<N, T> decompressed = mgard_cuda::decompress(handle, in_array);

    mgard_cuda::cudaFreeHostHelper(in_array_cpu);
    return pressio_data::copy(pressio_dtype_from_type<T>(), decompressed.getDataHost(), shape);
  }

  
  std::vector<size_t> const& shape;
  mgard_cuda::Config& config;
};

}

enum class pressio_mgard_mode {
  cpu = 0,
  gpu = 1
};

class mgard10_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    if(s == std::numeric_limits<double>::infinity()) {
      set(options, "pressio:abs", tolerance);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }
    set(options, "mgard:s", s);
    set(options, "mgard:tolerance", tolerance);
    set(options, "mgard:coordinates", coordinates);
    set(options, "mgard:norm_of_qoi", norm_of_qoi);
    set_type(options, "mgard:qoi_double", pressio_option_userptr_type);
    set_type(options, "mgard:qoi_float", pressio_option_userptr_type);
    set_type(options, "mgard:qoi_double_void", pressio_option_userptr_type);
    set_type(options, "mgard:qoi_float_void", pressio_option_userptr_type);
    set_type(options, "mgard:qoi_use_metric", pressio_option_int32_type);
    set_type(options, "mgard:qoi_metric_name", pressio_option_charptr_type);
    set(options, "mgard:execution_mode", execution_mode);
    set_type(options, "mgard:execution_mode_str", pressio_option_charptr_type);
    set(options, "mgard:cuda:dev_id", config.dev_id);
    set(options, "mgard:cuda:timing", static_cast<int32_t>(config.timing));
    set(options, "mgard:cuda:lossless", static_cast<int32_t>(config.lossless));

    set(options, "mgard:cuda:l_target", config.l_target);
    set(options, "mgard:cuda:huff_dict_size", config.huff_dict_size);
    set(options, "mgard:cuda:huff_block_size", config.huff_block_size);
    set(options, "mgard:cuda:lz4_block_size", config.lz4_block_size);
    set(options, "mgard:cuda:sync_and_check_all_kernels", static_cast<int32_t>(config.sync_and_check_all_kernels));
    set(options, "mgard:cuda:profile_kernels", static_cast<int32_t>(config.profile_kernels));
    set(options, "mgard:cuda:reduce_memory_footprint", static_cast<int32_t>(config.reduce_memory_footprint));
    set(options, "mgard:cuda:uniform_coord_mode", static_cast<int32_t>(config.uniform_coord_mode));

    set_type(options, "mgard:cuda:lossless_str", pressio_option_charptr_type);
    set(options, "mgard:cuda:error_bound_type", error_bound_type);
    set_type(options, "mgard:cuda:error_bound_type_str", pressio_option_charptr_type);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_single));
    set(options, "pressio:stability", "experimental");
    set(options, "mgard:execution_mode_str", std::vector<std::string>{"cpu", "gpu", "cuda"});
    set(options, "mgard:cuda:lossless_str", std::vector<std::string>{"cpu_lossless", "gpu_huffman", "gpu_huffman_lz4"});
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(MGARD is a error bounded lossy compressor based on using multi-level grids.
      More information can be found on onis [project homepage](https://github.com/CODARcode/MGARD))");
    set(options, "mgard:s", "the shape parameter");
    set(options, "mgard:tolerance", "the tolerance parameter");
    set(options, "mgard:coordinates", "coordinates for the tensor mesh hierarchy");
    set(options, "mgard:norm_of_qoi", "scaling factor determined by using MGARD-QOI");
    set(options, "mgard:qoi_double", "a pointer with type double(*)(int,int,int, double*) use for QOI mode");
    set(options, "mgard:qoi_float", "a pointer with type float(*)(int,int,int, float*) use for QOI mode");
    set(options, "mgard:qoi_double_void", "a pointer with type double(*)(int,int,int, double*, void*) use for QOI mode");
    set(options, "mgard:qoi_float_void", "a pointer with type float(*)(int,int,int, float*, void*) use for QOI mode");
    set(options, "mgard:qoi_use_metric", "1 to use pressio metrics for QOI");
    set(options, "mgard:qoi_metric_name", "name of the libpressio metric to retrieve");
    set(options, "mgard:execution_mode", "the execution_mode");
    set(options, "mgard:execution_mode_str", "the execution_mode as a human readable string");
    set(options, "mgard:cuda:dev_id", "device id for cuda execution");
    set(options, "mgard:cuda:timing", "timing support mgard cuda");
    set(options, "mgard:cuda:lossless", "which lossless compressors to use for mgard_cuda");
    set(options, "mgard:cuda:lossless_str", "which lossless compressors to use for mgard_cuda as a human readable string");
    set(options, "mgard:cuda:error_bound_type", "which error bound mode type to apply");
    set(options, "mgard:cuda:error_bound_type_str", "which error bound mode type to apply as a human readable string");

    set(options, "mgard:cuda:l_target", "mgard cuda's undocumented l_target parameter");
    set(options, "mgard:cuda:huff_dict_size",  "mgard cuda's undocumented huff_dict_size parameter");
    set(options, "mgard:cuda:huff_block_size",  "mgard cuda's undocumented huff_block_size parameter");
    set(options, "mgard:cuda:lz4_block_size",  "mgard cuda's undocumented lz4_block_size parameter");
    set(options, "mgard:cuda:uniform_coord_mode",  "mgard cuda's undocumented uniform_coord_mode parameter");
    set(options, "mgard:cuda:sync_and_check_all_kernels",  "mgard cuda's undocumented sync_and_check_all_kernels parameter");
    set(options, "mgard:cuda:profile_kernels",  "mgard cuda's undocumented profile_kernels parameter");
    set(options, "mgard:cuda:reduce_memory_footprint",  "mgard cuda's undocumented reduce_memory_footprint parameter");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    double abs;
    if(get(options, "pressio:abs", &abs) == pressio_options_key_set) {
      s = std::numeric_limits<double>::infinity();
      tolerance = abs;
    }
    get(options, "mgard:s", &s);
    get(options, "mgard:tolerance", &tolerance);
    get(options, "mgard:coordinates", &coordinates);
    get(options, "mgard:cuda:dev_id", &config.dev_id);
    std::string error_bound_type_str;
    if(get(options, "mgard:error_bound_type_str", &error_bound_type_str) == pressio_options_key_set) {
      if(error_bound_type_str == "abs") {
        error_bound_type = static_cast<int32_t>(mgard_cuda::error_bound_type::ABS);
      } else if(error_bound_type_str == "rel") {
        error_bound_type = static_cast<int32_t>(mgard_cuda::error_bound_type::REL);
      } else {
        return set_error(5, "unsupported error_bound_type_str " + error_bound_type_str);
      }
    } else {
      get(options, "mgard:cuda:error_bound_type", &error_bound_type);
    }
    std::string lossless_str;
    if(get(options, "mgard:cuda:lossless_str", &lossless_str) == pressio_options_key_set) {
      if(lossless_str == "cpu_lossless") {
        config.lossless = mgard_cuda::lossless_type::CPU_Lossless;
      } else if(lossless_str == "gpu_huffman") {
        config.lossless = mgard_cuda::lossless_type::GPU_Huffman;
      } else if(lossless_str == "gpu_huffman_lz4") {
        config.lossless = mgard_cuda::lossless_type::GPU_Huffman_LZ4;
      } else {
        return set_error(4, "unknown lossless compressor " + lossless_str);
      }
    } else {
      int32_t lossless;
      if(get(options, "mgard:cuda:lossless", &lossless) == pressio_options_key_set) {
        config.lossless = static_cast<mgard_cuda::lossless_type>(lossless);
      }
    }
    int32_t timing;
    if(get(options, "mgard:cuda:timing", &timing) == pressio_options_key_set) {
      config.timing = timing;
    }
    std::string execution_mode_str;
    if(get(options, "mgard:execution_mode_str", &execution_mode_str) == pressio_options_key_set) {
      if (execution_mode_str == "cpu") {
        execution_mode = static_cast<int32_t>(pressio_mgard_mode::cpu);
      } else if(execution_mode_str == "cuda" || execution_mode_str == "gpu") {
        execution_mode = static_cast<int32_t>(pressio_mgard_mode::gpu);
      } else {
        return set_error(3, "unsupported execution_mode_str " + execution_mode_str);
      }
    } else {
      get(options, "mgard:execution_mode", &execution_mode);
    }
    void* tmp=nullptr;
    void* data=nullptr;
    int use_metric_tmp = 0;
    if(get(options, "mgard:qoi_double", &tmp) == pressio_options_key_set) {
      caller_func = qoi::caller_ptr(compat::make_unique<qoi::call_cfunc<double>>(reinterpret_cast<double (*)(int,int,int, const double*)>(tmp)));
      recompute_qoi = true;
    } else if(get(options, "mgard:qoi_float", &tmp) == pressio_options_key_set) {
      caller_func = qoi::caller_ptr(compat::make_unique<qoi::call_cfunc<float>>(reinterpret_cast<float (*)(int,int,int, const float*)>(tmp)));
      recompute_qoi = true;
    } else if(get(options, "mgard:qoi_data", &data) == pressio_options_key_set) {
      if(get(options, "mgard:qoi_double_void", &tmp) == pressio_options_key_set) {
        caller_func = qoi::caller_ptr(compat::make_unique<qoi::call_cfuncv<double>>(reinterpret_cast<double (*)(int,int,int, const double*, void*)>(tmp), data));
        recompute_qoi = true;
      } else if(get(options, "mgard:qoi_float_void", &tmp) == pressio_options_key_set) {
        caller_func = qoi::caller_ptr(compat::make_unique<qoi::call_cfuncv<float>>(reinterpret_cast<float (*)(int,int,int, const float*, void*)>(tmp), data));
        recompute_qoi = true;
      }
    } else if(get(options, "mgard:qoi_use_metric", &use_metric_tmp) == pressio_options_key_set) {
      std::string metric_name;
      if(use_metric_tmp && get(options, "mgard:qoi_metric_name", &metric_name) == pressio_options_key_set) {
        caller_func = qoi::caller_ptr(compat::make_unique<qoi::call_pressio_metric>(metric_name, get_metrics(),pressio_byte_dtype));
        recompute_qoi = true;
      }
    } else if(get(options, "mgard:qoi_reset", &use_metric_tmp) == pressio_options_key_set) {
      caller_func = qoi::caller_ptr{};
      recompute_qoi = true;
    }

    get(options, "mgard:cuda:l_target", &config.l_target);
    get(options, "mgard:cuda:huff_dict_size", &config.huff_dict_size);
    get(options, "mgard:cuda:huff_block_size", &config.huff_block_size);
    get(options, "mgard:cuda:lz4_block_size", &config.lz4_block_size);
    get(options, "mgard:cuda:uniform_coord_mode", &config.uniform_coord_mode);
    if(get(options, "mgard:cuda:sync_and_check_all_kernels", &use_metric_tmp) == pressio_options_key_set) {
      config.sync_and_check_all_kernels = use_metric_tmp;
    }
    if(get(options, "mgard:cuda:profile_kernels", &use_metric_tmp) == pressio_options_key_set)  {
      config.profile_kernels = use_metric_tmp;
    }
    if(get(options, "mgard:cuda:reduce_memory_footprint", &use_metric_tmp) == pressio_options_key_set) {
      config.reduce_memory_footprint = use_metric_tmp;
    }

    
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    if(last_shape != input->dimensions()) {
      recompute_qoi = true;
    }
    last_shape = input->dimensions();
    try {
    if(execution_mode == static_cast<int32_t>(pressio_mgard_mode::cpu)) {
      auto compress = cpu::compress_operator{
        coordinates, input->dimensions(), s,
          tolerance, norm_of_qoi, caller_func,
          recompute_qoi, norm_time};
      if(input->dtype() == pressio_float_dtype) {
        *output = compress(static_cast<float*>(input->data()), static_cast<float*>(input->data()) + input->num_elements());
      } else if (input->dtype() == pressio_double_dtype) {
        *output = compress(static_cast<double*>(input->data()), static_cast<double*>(input->data()) + input->num_elements());
      } else {
        return set_error(1, "type not supported");
      }
    } else if(execution_mode == static_cast<int32_t>(pressio_mgard_mode::gpu)) {
      auto compress = gpu::compress_operator{
        coordinates, input->dimensions(), s,
          tolerance, norm_of_qoi, caller_func,
          recompute_qoi, norm_time, config};
      if(input->dtype() == pressio_float_dtype) {
        *output = compress(static_cast<float*>(input->data()), static_cast<float*>(input->data()) + input->num_elements());
      } else if (input->dtype() == pressio_double_dtype) {
        *output = compress(static_cast<double*>(input->data()), static_cast<double*>(input->data()) + input->num_elements());
      } else {
        return set_error(1, "type not supported");
      }
    } else {
        return set_error(2, "mode not supported");
    }
    } catch(std::domain_error const& ex) {
      return set_error(3, ex.what());
    }
    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    if(execution_mode == static_cast<int32_t>(pressio_mgard_mode::cpu)) {
      auto decompress = cpu::decompress_operator{output->dimensions()};
      if(output->dtype() == pressio_float_dtype) {
        *output = decompress.operator()<float>(static_cast<unsigned char*>(input->data()), static_cast<unsigned char*>(input->data()) + input->num_elements());
      } else if (output->dtype() == pressio_double_dtype) {
        *output = decompress.operator()<double>(static_cast<unsigned char*>(input->data()), static_cast<unsigned char*>(input->data()) + input->num_elements());
      } else {
        return set_error(1, "type not supported");
      }
    } else if(execution_mode == static_cast<int32_t>(pressio_mgard_mode::gpu)) {
      auto decompress = gpu::decompress_operator{output->dimensions(), config};
      if(output->dtype() == pressio_float_dtype) {
        *output = decompress.operator()<float>(static_cast<unsigned char*>(input->data()), static_cast<unsigned char*>(input->data()) + input->num_elements());
      } else if (output->dtype() == pressio_double_dtype) {
        *output = decompress.operator()<double>(static_cast<unsigned char*>(input->data()), static_cast<unsigned char*>(input->data()) + input->num_elements());
      } else {
        return set_error(1, "type not supported");
      }
    } else {
        return set_error(2, "mode not supported");
    }
    return 0;
  }

  int major_version() const override { return MGARD_VERSION_MAJOR; }
  int minor_version() const override { return MGARD_VERSION_MINOR; }
  int patch_version() const override { return MGARD_VERSION_PATCH; }
  const char* version() const override { return MGARD_VERSION_STR; }
  const char* prefix() const override { return "mgard"; }

  pressio_options get_metrics_results_impl() const override {
    pressio_options opts;
    set(opts, "mgard:norm_of_qoi", norm_of_qoi);
    set(opts, "mgard:norm_time", norm_time);

    return opts;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<mgard10_compressor_plugin>(*this);
  }

  compat::optional<pressio_data> coordinates{}; 
  double s = std::numeric_limits<double>::infinity(), tolerance = 1e-6;
  compat::optional<double> norm_of_qoi;
  qoi::caller_ptr caller_func;
  bool recompute_qoi = true;
  std::vector<size_t> last_shape;
  compat::optional<uint64_t> norm_time;
  int32_t execution_mode = static_cast<int32_t>(pressio_mgard_mode::cpu);
  int32_t error_bound_type = static_cast<int32_t>(mgard_cuda::error_bound_type::ABS);
  mgard_cuda::Config config = mgard_cuda::Config();
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "mgard", []() {
  return compat::make_unique<mgard10_compressor_plugin>();
});

} }
