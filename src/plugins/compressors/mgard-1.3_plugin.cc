#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "cleanup.h"
#include <sstream>
#include <mgard/compress_x.hpp>
#include <mgard/MGARDConfig.hpp>
#if MGARD_ENABLE_OPENMP
#include <omp.h>
#endif

namespace libpressio { namespace mgard_ns {


  template <class Enum>
  std::vector<std::string> to_names(std::map<Enum, std::string> const& m) {
    std::vector<std::string> ret;
    std::transform(m.begin(), m.end(), std::back_inserter(ret), [](typename std::map<Enum, std::string>::const_reference it) {
        return it.second;
    });
    return ret;
  }
  template <class Enum>
  Enum find_string(std::map<Enum, std::string> const& m, std::string const& str, std::string const& type) {
    auto it = std::find_if(m.begin(), m.end(), [&str](typename std::map<Enum, std::string>::const_reference it){
        return it.second == str;
    });
    if(it != m.end()) {
      return it->first;
    } else {
      throw std::runtime_error(type + " key not found " + str);
    }
  }
  static std::map<mgard_x::lossless_type, std::string> lossless_types {
    {mgard_x::lossless_type::Huffman, "huffman"},
    { mgard_x::lossless_type::Huffman_LZ4, "huffman_lz4"},
    { mgard_x::lossless_type::Huffman_Zstd, "huffman_zstd"},
    { mgard_x::lossless_type::CPU_Lossless, "cpu_lossless"},
  };
  static std::map<mgard_x::decomposition_type, std::string> decomp_types {
    {mgard_x::decomposition_type::MultiDim, "multidim"},
    {mgard_x::decomposition_type::SingleDim, "singledim"},
  };
  static std::map<mgard_x::error_bound_type, std::string> bound_types {
    {mgard_x::error_bound_type::ABS, "abs"},
      { mgard_x::error_bound_type::REL, "rel"}
  };
  static std::map<mgard_x::device_type, std::string> dev_types {
    {mgard_x::device_type::AUTO, "auto"},
      { mgard_x::device_type::SERIAL, "serial"},
      { mgard_x::device_type::OPENMP, "openmp"},
      { mgard_x::device_type::CUDA, "cuda"},
      { mgard_x::device_type::HIP, "hip"},
      { mgard_x::device_type::SYCL, "sycl"},
      { mgard_x::device_type::NONE, "none"}
  };

class mgard_compressor_plugin : public libpressio_compressor_plugin {
  std::string mgard_decomposition_to_string(mgard_x::decomposition_type d_type) const {
    return decomp_types.at(d_type);
  }
  std::string mgard_dev_type_to_string(mgard_x::device_type dev_type) const {
    return dev_types.at(dev_type);
  }
  std::string mgard_bound_to_string(mgard_x::error_bound_type b_type) const {
    return bound_types.at(b_type);
  }
  std::string mgard_lossless_to_string(mgard_x::lossless_type l_type) const {
    return lossless_types.at(l_type);
  }
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;

    if(s == std::numeric_limits<double>::infinity() && bound_type == mgard_x::error_bound_type::ABS) {
      set(options, "pressio:abs", tolerance);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }
    set(options, "pressio:nthreads", nthreads);
    set(options, "mgard:nthreads", nthreads);
    set(options, "mgard:dev_type", static_cast<uint8_t>(config.dev_type));
    set(options, "mgard:dev_type_str", mgard_dev_type_to_string(config.dev_type));
    set(options, "mgard:dev_id", config.dev_id);
    set(options, "mgard:decomposition", static_cast<uint8_t>(config.decomposition));
    set(options, "mgard:decomposition_str", mgard_decomposition_to_string(config.decomposition));
    set(options, "mgard:huff_dict_size", config.huff_dict_size);
    set(options, "mgard:huff_block_size", config.huff_block_size);
    set(options, "mgard:lz4_block_size", config.lz4_block_size);
    set(options, "mgard:zstd_compress_level", config.zstd_compress_level);
    set(options, "mgard:normalize_coordinates", config.normalize_coordinates);
    set(options, "mgard:lossless_type", static_cast<uint8_t>(config.lossless));
    set(options, "mgard:lossless_type_str", mgard_lossless_to_string(config.lossless));
    set(options, "mgard:reorder", config.reorder);
    set(options, "mgard:log_level", config.log_level);
    set(options, "mgard:s", s);
    set(options, "mgard:tolerance", tolerance);
    set(options, "mgard:error_bound_type", static_cast<uint8_t>(bound_type));
    set(options, "mgard:error_bound_type_str", mgard_bound_to_string(bound_type));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "mgard:dev_type_str", to_names(dev_types));
    set(options, "mgard:decomposition_str", to_names(decomp_types));
    set(options, "mgard:lossless_type_str", to_names(lossless_types));
    set(options, "mgard:error_bound_type_str", to_names(bound_types));
    set(options, "mgard:serial_enabled", MGARD_ENABLE_SERIAL);
    set(options, "mgard:openmp_enabled", MGARD_ENABLE_OPENMP);
    set(options, "mgard:cuda_enabled", MGARD_ENABLE_CUDA);
    set(options, "mgard:hip_enabled", MGARD_ENABLE_HIP);
    set(options, "mgard:sycl_enabled", MGARD_ENABLE_SYCL);
    set(options, "pressio:stability", "experimental");
    
        std::vector<std::string> invalidations {"mgard:huff_dict_size", "mgard:huff_block_size", "mgard:lz4_block_size", "mgard:zstd_compress_level", "mgard:normalize_coordinates", "mgard:reorder", "mgard:log_level", "mgard:s", "mgard:tolerance", "pressio:abs", "mgard:decomposition", "mgard:decomposition_str", "mgard:lossless_type", "mgard:lossless_type_str", "mgard:error_bound_type", "mgard:error_bound_type_str"}; 
        std::vector<std::string> runtime_invalidations {"mgard:huff_dict_size", "mgard:dev_id", "mgard:huff_block_size", "mgard:lz4_block_size", "mgard:zstd_compress_level", "mgard:normalize_coordinates", "mgard:reorder", "mgard:log_level", "mgard:s", "mgard:tolerance", "pressio:abs", "pressio:nthreads", "mgard:nthreads", "mgard:dev_type", "mgard:dev_type_str", "mgard:decomposition", "mgard:decomposition_str", "mgard:lossless_type", "mgard:lossless_type_str", "mgard:error_bound_type", "mgard:error_bound_type_str"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, runtime_invalidations));

    
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:abs", "pressio:nthreads"}));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(MGARD+

MGARD (MultiGrid Adaptive Reduction of Data) is a technique for multilevel lossy compression of scientific data based on the theory of multigrid methods.

MGARD's theoretical foundation and software implementation are discussed in the following papers.
Reference [2] covers the simplest case and is a natural starting point.
Reference [6] covers the design and implementation on GPU heterogeneous systems.

1. Ben Whitney. [Multilevel Techniques for Compression and Reduction of Scientific Data.][thesis] PhD thesis, Brown University, 2018.
2. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Univariate Case.][univariate] *Computing and Visualization in Science* 19, 65–76, 2018.
3. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Multivariate Case.][multivariate] *SIAM Journal on Scientific Computing* 41 (2), A1278–A1303, 2019.
4. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—Quantitative Control of Accuracy in Derived Quantities.][quantities] *SIAM Journal on Scientific Computing* 41 (4), A2146–A2171, 2019.
5. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Unstructured Case.][unstructured] *SIAM Journal on Scientific Computing*, 42 (2), A1402–A1427, 2020.
6. Jieyang Chen et al. [Accelerating Multigrid-based Hierarchical Scientific Data Refactoring on GPUs.][gpu] *35th IEEE International Parallel & Distributed Processing Symposium*, May 17–21, 2021.

[thesis]: https://doi.org/10.26300/ya1v-hn97
[univariate]: https://doi.org/10.1007/s00791-018-00303-9
[multivariate]: https://doi.org/10.1137/18M1166651
[quantities]: https://doi.org/10.1137/18M1208885
[unstructured]: https://doi.org/10.1137/19M1267878
[gpu]: https://arxiv.org/pdf/2007.04457
    )");
    set(options, "mgard:nthreads", "number of CPU threads when using dev_type_str==\"openmp\"");
    set(options, "mgard:dev_type", "device execution type");
    set(options, "mgard:dev_type_str", "device execution type name");
    set(options, "mgard:dev_id", "execution device id");
    set(options, "mgard:decomposition", "decomposition type");
    set(options, "mgard:decomposition_str", "decomposition type name");
    set(options, "mgard:huff_dict_size", "huffman dictionary size");
    set(options, "mgard:huff_block_size", "huffman block size");
    set(options, "mgard:lz4_block_size", "lz4 block size");
    set(options, "mgard:zstd_compress_level", "Zstandard compress level");
    set(options, "mgard:normalize_coordinates", "normalize_coordinates");
    set(options, "mgard:lossless_type", "what lossless encoding to use");
    set(options, "mgard:lossless_type_str", "name of lossless encoding to use");
    set(options, "mgard:reorder", "reordering dims");
    set(options, "mgard:log_level", "logging level");
    set(options, "mgard:s", "shape parameter");
    set(options, "mgard:tolerance", "tolerance level");
    set(options, "mgard:error_bound_type", "error bound type");
    set(options, "mgard:error_bound_type_str", "error_bound_type name");
    set(options, "mgard:serial_enabled", "mgard with built with serial execution");
    set(options, "mgard:openmp_enabled", "mgard was built with openmp execution");
    set(options, "mgard:cuda_enabled", "mgard was built with cuda execution");
    set(options, "mgard:hip_enabled", "mgard was built with hip execution");
    set(options, "mgard:sycl_enabled", "mgard was built with sycl execution");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    if(get(options, "pressio:abs", &tolerance) == pressio_options_key_set) {
      s = std::numeric_limits<double>::infinity();
      bound_type = mgard_x::error_bound_type::ABS;
    }
    {
      uint32_t tmp;
      if(get(options, "pressio:nthreads", &tmp) == pressio_options_key_set){
        if(tmp > 0) {
          config.dev_type = mgard_x::device_type::OPENMP;
          nthreads = tmp;
        }
        else return set_error(1, "nthreads must be positive");
      }
      if(get(options, "mgard:nthreads", &tmp) == pressio_options_key_set){
        if(tmp > 0) {
          nthreads = tmp;
        }
        else return set_error(1, "nthreads must be positive");
      }
    }
    uint8_t tmp;
    if(get(options, "mgard:dev_type", &tmp) == pressio_options_key_set)  {
      config.dev_type = static_cast<mgard_x::device_type>(tmp);
    }

    std::string tmp_str;
    if(get(options, "mgard:dev_type_str", &tmp_str) == pressio_options_key_set){
        config.dev_type = find_string(dev_types, tmp_str, "device type");
    }
    get(options, "mgard:dev_id", &config.dev_id);
    if(get(options, "mgard:decomposition", &tmp) == pressio_options_key_set) {
      config.decomposition = static_cast<mgard_x::decomposition_type>(tmp);
    }

    if(get(options, "mgard:decomposition_str", &tmp_str) == pressio_options_key_set) {
      config.decomposition = find_string(decomp_types, tmp_str, "decomposition type");
    }
    get(options, "mgard:huff_dict_size", &config.huff_dict_size);
    get(options, "mgard:huff_block_size", &config.huff_block_size);
    get(options, "mgard:lz4_block_size", &config.lz4_block_size);
    get(options, "mgard:zstd_compress_level", &config.zstd_compress_level);
    get(options, "mgard:normalize_coordinates", &config.normalize_coordinates);
    if(get(options, "mgard:lossless_type", &tmp) == pressio_options_key_set) {
      config.lossless = static_cast<mgard_x::lossless_type>(tmp);
    }
    if(get(options, "mgard:lossless_type_str", &tmp_str) == pressio_options_key_set) {
      config.lossless = find_string(lossless_types, tmp_str, "lossless");
    }
    get(options, "mgard:reorder", &config.reorder);
    get(options, "mgard:log_level", &config.log_level);
    get(options, "mgard:s", &s);
    get(options, "mgard:tolerance", &tolerance);
    if(get(options, "mgard:error_bound_type", &tmp) == pressio_options_key_set) {
      bound_type = static_cast<mgard_x::error_bound_type>(tmp);
    }
    if(get(options, "mgard:error_bound_type_str", &tmp_str) == pressio_options_key_set) {
      bound_type = find_string(bound_types, tmp_str, "error_bound");
    }
    return 0;
  }

  mgard_x::data_type to_mgard_type(pressio_dtype dtype) {
    if(dtype == pressio_float_dtype) return mgard_x::data_type::Float;
    else if(dtype == pressio_double_dtype) return mgard_x::data_type::Double;
    else {
      std::stringstream ss;
      ss << "unsupported dtype " << dtype;
      throw std::runtime_error(ss.str());
    }
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
#if MGARD_ENABLE_OPENMP
      int save_threads = omp_get_num_threads();
      omp_set_num_threads(nthreads);
      cleanup restore_threads([save_threads]{omp_set_num_threads(save_threads);});
#endif
      void* output_data = output->data();
      size_t compressed_size;
      bool pre_allocated = output->has_data();
      auto const& dims = input->normalized_dims();
      mgard_x::compress(dims.size(), to_mgard_type(input->dtype()), dims, tolerance, s,
                      bound_type, input->data(),
                      output_data, compressed_size, config, pre_allocated);

      if(pre_allocated) {
      output->set_dimensions({compressed_size});
      output->set_dtype(pressio_byte_dtype);
      } else {
        *output = pressio_data::move(
            pressio_byte_dtype,
            output_data,
            {compressed_size},
            pressio_data_libc_free_fn,
            nullptr
            );
      }
      return 0;
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
#if MGARD_ENABLE_OPENMP
      int save_threads = omp_get_num_threads();
      omp_set_num_threads(nthreads);
      cleanup restore_threads([save_threads]{omp_set_num_threads(save_threads);});
#endif
    void* output_data = output->data();
    bool pre_allocated = output->has_data();
    mgard_x::decompress(input->data(), input->size_in_bytes(),
                        output_data, config, pre_allocated);
    if(!pre_allocated) {
      *output = pressio_data::move(
          output->dtype(),
          output_data,
          output->dimensions(),
          pressio_data_libc_free_fn,
          nullptr
          );
    }
    return 0;
  }

  int major_version() const override { return MGARD_VERSION_MAJOR; }
  int minor_version() const override { return MGARD_VERSION_MINOR; }
  int patch_version() const override { return MGARD_VERSION_PATCH; }
  const char* version() const override { return MGARD_VERSION_STR; }
  const char* prefix() const override { return "mgard"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<mgard_compressor_plugin>(*this);
  }

  uint32_t nthreads = 1;
  double tolerance = 1e-4, s=std::numeric_limits<double>::infinity();
  mgard_x::error_bound_type bound_type = mgard_x::error_bound_type::ABS;
  mgard_x::Config config = []{
    //prefer serial on the CPU if possible
    mgard_x::Config config;
    if(MGARD_ENABLE_SERIAL) {
      config.dev_type = mgard_x::device_type::SERIAL;
    } else if(MGARD_ENABLE_OPENMP) {
      config.dev_type = mgard_x::device_type::OPENMP;
    } else {
      config.dev_type = mgard_x::device_type::AUTO;
    }
    return config;
  }();
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "mgard", []() {
  return compat::make_unique<mgard_compressor_plugin>();
});

} }
