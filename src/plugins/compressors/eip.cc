
#include <array>
#include <cuda_runtime.h>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "cusz/config/eip.hh"
#include "cusz/kernel/predictor.hh"
#include "cusz/kernel/spv.hh"
#include "cusz/mem/cxx_backends.h"
#include "cusz/hf_hl.hh"
#include "cusz/mem/compbuf_pbk.hh"

namespace libpressio { namespace compressors { namespace eip_ns {


    constexpr size_t Radius = 128;
using T = float; using M = uint32_t;
using E = uint16_t;
template <class T> using PC_ref = psz::PredConfig<T, psz::PredFunc<Toggle::ZigZagDisabled>>;
template <class T> using GPU_xlrz_ref = psz::module::GPU_x_lorenzo_nd<T, PC_ref<T>>;
using Buf = psz::CompressorBufferPbk2<float, 128>;
using PC_eip = psz::PredConfig<
        T, psz::PredFunc<
           Toggle::ZigZagDisabled, Toggle::StatLocalEnabled, Toggle::StatGlobalDisabled,
           Toggle::QuantGroupingDisabled, Toggle::FutureEIPEnabled>>;
using GPU_EIP_comp_r128 = psz::module::GPU_c_lorenzo_1d_eip<f4, EIP_PC_f4, 128, 4 /* 4 points per thread, best case for perf*/>;
using BufToggle = psz::CompressorBufferPbkToggle;

BufToggle toggle_eip{.pbk_all = true};

class eip_compressor_plugin : public libpressio_compressor_plugin {

private:
  std::unique_ptr<Buf> eip;
  std::array<size_t, 3> dims;
  double bound = 1e-4;
public:
  eip_compressor_plugin(): libpressio_compressor_plugin(),  eip(std::make_unique<Buf>(1, 1, 1, false /* will revise in the future*/, &toggle_eip)),  dims{1,1,1} {
  }
  eip_compressor_plugin(eip_compressor_plugin const& rhs): libpressio_compressor_plugin(rhs), eip(std::make_unique<Buf>(rhs.dims[0], rhs.dims[1], rhs.dims[2], false /* will revise in the future*/, &toggle_eip)), dims(rhs.dims) {
    // Buffer size need to be deteremined on intialization.
    eip = std::make_unique<Buf>(1, 1, 1, false /* will revise in the future*/, &toggle_eip);
  }
  eip_compressor_plugin& operator=(eip_compressor_plugin const& rhs){
    // Buffer size need to be deteremined on intialization.
    dims = rhs.dims;
    eip = std::make_unique<Buf>(dims[0], dims[1], dims[2], false /* will revise in the future*/, &toggle_eip);
    return *this;
  }


  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:abs", bound);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> invalidations {}; 
    std::vector<pressio_configurable const*> invalidation_children {}; 
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{}));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"()");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:abs", &bound);
    return 0;
  }


  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
    if(real_input->dtype() != pressio_float_dtype) return set_error(1, "unsupported type");
    auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);

    auto i_dims = input.normalized_dims(3,1);
    if(!std::equal(i_dims.begin(), i_dims.end(), dims.begin())) {
        eip = std::make_unique<Buf>(i_dims[0], i_dims[1], i_dims[2], false /* will revise in the future*/, &toggle_eip);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    GPU_EIP_comp_r128::kernel(
          (float*)input.data(), input.num_elements(), (void*)eip->outlier(), bound, eip->pbk_books(),
          eip->pbk_book_IDs(), eip->pbk_bitstream(), eip->pbk_bits(),
          eip->pbk_entries(), eip->pbk_loc(), eip->pbk_break(), stream);

    auto endloc = eip->pbk_encoding_endloc(); // contains a d2h copy of one size_t
    auto brlen = eip->host_get_pbk_break_num();
    auto splen = eip->host_get_unpredictable_num();

    // right now, the errno is arbirarily defined.
    std::stringstream ss;
    if (brlen >= eip->pbk_break_num_max() - 1) {
      ss << "[psz::warning::pbk_run::EIP] max allowed enc-breaking ("
           << eip->pbk_break_num_max() << ") exceeded";
      return set_error(1, ss.str());
    }
    if (splen >= eip->unpredictable_num_max() - 1) {
      ss << "[psz::warning::pbk_run::EIP] max allowed unpredictable ("
           << eip->outlier_ratio() << " * input-len) exceeded";
      return set_error(1, ss.str());
    }

    *output = pressio_data::join({
            pressio_data(std::make_tuple(endloc, brlen, splen)),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_book_IDs())>>(), eip->pbk_book_IDs(), {eip->num_chunk()}, "cudamalloc"),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_entries())>>(), eip->pbk_entries(), {eip->num_chunk()}, "cudamalloc"),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_bits())>>(), eip->pbk_bits(), {eip->num_chunk()}, "cudamalloc"),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_bitstream())>>(), eip->pbk_bitstream(), {endloc}, "cudamalloc"),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_break_val())>>(), eip->pbk_break_val(), {brlen}, "cudamalloc"),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_break_idx())>>(), eip->pbk_break_idx(), {brlen}, "cudamalloc"),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->unpredictable_val())>>(), eip->unpredictable_val(), {splen}, "cudamalloc"),
            pressio_data::nonowning(pressio_dtype_from_type<std::decay_t<decltype(*eip->unpredictable_idx())>>(), eip->unpredictable_idx(), {splen}, "cudamalloc"),
            });


    return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* output) override
  {
    auto input = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *real_input);

    auto i_dims = output->normalized_dims(3,1);
    if(!std::equal(i_dims.begin(), i_dims.end(), dims.begin())) {
        eip = std::make_unique<Buf>(i_dims[0], i_dims[1], i_dims[2], false /* will revise in the future*/, &toggle_eip);
    }


    std::vector<pressio_data> restore{
            pressio_data(),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_book_IDs())>>(),  domain_plugins().build("cudamalloc")),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_entries())>>(), domain_plugins().build( "cudamalloc")),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_bits())>>(), domain_plugins().build( "cudamalloc")),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_bitstream())>>(),  domain_plugins().build( "cudamalloc")),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_break_val())>>(),  domain_plugins().build( "cudamalloc")),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->pbk_break_idx())>>(),  domain_plugins().build( "cudamalloc")),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->unpredictable_val())>>(), domain_plugins().build( "cudamalloc")),
            pressio_data::type_domain(pressio_dtype_from_type<std::decay_t<decltype(*eip->unpredictable_idx())>>(), domain_plugins().build( "cudamalloc")),
    };
    pressio_data::split(input, restore);
    auto metadata = std::move(restore.at(0));
    auto book_ids = std::move(restore.at(1));
    auto entries = std::move(restore.at(2));
    auto bits = std::move(restore.at(3));
    auto bitstream = std::move(restore.at(4));
    auto break_val = std::move(restore.at(5));
    auto break_idx = std::move(restore.at(6));
    auto unpred_val = std::move(restore.at(7));
    auto unpred_idx = std::move(restore.at(8));
    *output = domain_manager().make_writeable(domain_plugins().build("cudamalloc"), std::move(*output));

    // The following variables are from the compression archive:
    // len (the original #elements), splen, brlen)

    // Also, assume mem_eip buffer is reused.

    size_t splen, brlen, endloc;
    std::tie(endloc, brlen, splen) = restore.at(0).to_tuple<std::tuple<size_t, size_t, size_t>>();
    

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    if (splen)
      psz::spv_scatter_naive<CUDA, T, M>(
          (T*)unpred_val.data(), (uint32_t*)unpred_idx.data(), splen, (float*)output->data(),
          nullptr, stream);
    cudaStreamSynchronize(stream);

    size_t len = output->num_elements();
    pressio_data ectrl_eip(pressio_data::owning(pressio_dtype_from_type<E>(), {len}));
    phf::cuhip::modules<E, Hf>::CPU_pbk_coarse_decode(                                            
      (uint32_t*)bitstream.data(), endloc, eip->pbk_revbooks_h(), eip->PBK_RVBK_BYTES(), 
      (uint8_t*)book_ids.data(), (uint16_t*)bits.data(), (uint32_t*)entries.data(),               
      (E*)ectrl_eip.data(), len);

    // request an h2d copy: use host buffer for Huffman decoding
    ectrl_eip = domain_manager().make_readable(domain_plugins().build("cudamalloc"), std::move(ectrl_eip));

    // Then fix the breakings during parallel encoding
    if (brlen)
      psz::spv_scatter_naive<CUDA, E, M>(
          (uint16_t*)break_val.data(), (uint32_t*)break_idx.data(), brlen, (uint16_t*)ectrl_eip.data(), nullptr,
          stream);
    cudaStreamSynchronize(stream);

    auto d_space = (T*)output->data(), d_xdata = (T*)output->data();  // aliases

    std::array<size_t, 3> len3_std;
    GPU_xlrz_ref<T>::kernel(eip->ectrl(), d_space, d_xdata, len3_std, bound, Radius, stream);
    cudaStreamSynchronize(stream);



    return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "eip"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<eip_compressor_plugin>(*this);
  }

};

pressio_register registration(compressor_plugins(), "eip", []() {
  return compat::make_unique<eip_compressor_plugin>();
});

} } }

