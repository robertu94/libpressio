#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cusz/cusz_version.h>
#if __has_include(<cusz.h>)
#include <cusz.h>
#elif __has_include(<cusz/include/cusz.h>)
#include <cusz/include/cusz.h>
#else
#error "cuSZ header cusz.h not found"
#endif

namespace libpressio {
namespace compressors {
namespace cusz_ns {

class cusz_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override {
    struct pressio_options options;
    if (eb_mode == "abs") {
      set(options, "pressio:abs", err_bnd);
    } else {
      set_type(options, "pressio:abs", pressio_option_double_type);
    }

    if (eb_mode == "rel") {
      set(options, "pressio:rel", err_bnd);
    } else {
      set_type(options, "pressio:rel", pressio_option_double_type);
    }

    set(options, "cusz:mode_str", eb_mode);
    set(options, "cusz:bound", err_bnd);
    set(options, "cusz:histogram_type", histogram_type);
    set(options, "cusz:predictor_type", predictor_type);
    set(options, "cusz:codec1_type", codec1_type);
    set(options, "cusz:codec2_type", codec2_type);
    set_type(options, "cusz:reset", pressio_option_int32_type);

    return options;
  }

  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_serialized);
    set(options, "pressio:stability", "experimental");
    set(options, "cusz:mode_str", std::vector<std::string>{"abs", "rel"});
    set(options, "cusz:predictor_type",
        std::vector<std::string>{"lrz", "lrz-zz", "spl-y25", "spl-y24"});
    set(options, "cusz:histogram_type", std::vector<std::string>{"generic", "sparse", "null"});
    set(options, "cusz:codec1_type",
        std::vector<std::string>{"hf-rev2", "hfr-v2", "hfr-v4", "hfr-pbkc", "hfr-pbkgo", "lc-tcms",
                                 "lc-drh", "lc-bitr", "lc-rtr", "fzg"});
    set(options, "cusz:codec2_type", std::vector<std::string>{"none", "lc-bitr", "lc-rtr"});
    return options;
  }

  struct pressio_options get_documentation_impl() const override {
    struct pressio_options options;
    set(options, "pressio:description", R"(A GPU based implementation of SZ for Nvidia GPUs)");
    set(options, "cusz:mode_str", "error bound mode");
    set(options, "cusz:bound", "bound of the error bound");
    set(options, "cusz:predictor_type", "what predictor type to use");
    set(options, "cusz:histogram_type", "pipeline histogram stage");
    set(options, "cusz:codec1_type", "what compression codec is firstly used");
    set(options, "cusz:codec2_type", "what compression codec is secondly used");
    set(options, "cusz:reset",
        "write-only trigger: set to 1 to clear the reused compress/decompress state between runs");
    return options;
  }

  int set_options_impl(struct pressio_options const &options) override {
    if (get(options, "pressio:abs", &err_bnd) == pressio_options_key_set) {
      eb_mode = "abs";
    }
    if (get(options, "pressio:rel", &err_bnd) == pressio_options_key_set) {
      eb_mode = "rel";
    }

    get(options, "cusz:mode_str", &eb_mode);
    get(options, "cusz:bound", &err_bnd);
    get(options, "cusz:histogram_type", &histogram_type);
    get(options, "cusz:predictor_type", &predictor_type);
    get(options, "cusz:codec1_type", &codec1_type);
    get(options, "cusz:codec2_type", &codec2_type);

    int32_t do_reset = 0;
    if (get(options, "cusz:reset", &do_reset) == pressio_options_key_set and do_reset != 0)
      return reset();

    return 0;
  }

  // clear states between runs
  int reset() {
    if (not state)
      return 0;
    if (state->enc and state->enc_dirty) {
      int ec = psz_compress_reset(state->enc);
      if (ec != PSZ_SUCCESS)
        return psz_error(ec, "psz_compress_reset");
      state->enc_dirty = false;
    }
    if (state->dec) {
      int ec = psz_decompress_reset(state->dec);
      if (ec != PSZ_SUCCESS)
        return psz_error(ec, "psz_decompress_reset");
    }
    return 0;
  }

  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
      switch (input->dtype()) {
      case pressio_float_dtype:
        return compress_typed<float>(input, output);
      case pressio_double_dtype:
        return compress_typed<double>(input, output);
      default:
        return set_error(1, "unsupported dtype");
      }
    } catch (std::exception const &ex) {
      return set_error(2, ex.what());
    }
  }

  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
      switch (output->dtype()) {
      case pressio_float_dtype:
        return decompress_typed<float>(input, output);
      case pressio_double_dtype:
        return decompress_typed<double>(input, output);
      default:
        return set_error(1, "unsupported dtype");
      }
    } catch (std::exception const &ex) {
      return set_error(2, ex.what());
    }
  }

  psz_dtype to_psz_dtype(pressio_dtype dtype) {
    switch (dtype) {
    case pressio_float_dtype:
      return F4;
    case pressio_double_dtype:
      return F8;
    default: {
      std::stringstream ss;
      ss << "only float and double supported: " << dtype;
      throw std::domain_error(ss.str());
    }
    }
  }

  psz_predictor to_psz_predictor(std::string const &s) {
    if (s == "lrz")
      return Lorenzo;
    if (s == "lrz-zz")
      return LorenzoZigZag;
    if (s == "spl-y25")
      return SplineY25;
    if (s == "spl-y24")
      return SplineY24;
    throw std::domain_error("unsupported predictor_type: " + s);
  }

  // Codec-1: HFR-v3 and HFR-PBKF are out.
  psz_codec to_psz_codec1(std::string const &s) {
    if (s == "hf-rev2")
      return HF_r2;
    if (s == "hfr-v2")
      return HFR;
    if (s == "hfr-v4")
      return HFR_V4;
    if (s == "hfr-pbkc")
      return HFR_PBKC;
    if (s == "hfr-pbkgo")
      return HFR_PBKGO;
    if (s == "lc-tcms")
      return LC_TCMS;
    if (s == "lc-drh")
      return LC_DRH;
    if (s == "lc-bitr")
      return LC_BITR;
    if (s == "lc-rtr")
      return LC_RTR;
    if (s == "fzg")
      return FZG;
    throw std::domain_error("unsupported codec type: " + s);
  }

  // Codec-2: LC-* only.
  psz_codec to_psz_codec2(std::string const &s) {
    if (s == "none")
      return CodecNull;
    if (s == "lc-bitr")
      return LC_BITR;
    if (s == "lc-rtr")
      return LC_RTR;
    throw std::domain_error("unsupported codec type: " + s);
  }

  psz_hist to_psz_hist(std::string const &s) {
    if (s == "generic")
      return HistGeneric;
    if (s == "sparse")
      return HistSp;
    if (s == "null")
      return HistNull;
    throw std::domain_error("unsupported histogram_type: " + s);
  }

  int cuda_error(cudaError_t ec) { return set_error(12, cudaGetErrorString(ec)); }

  int psz_error(int ec, char const *where) {
    if (ec == PSZ_SUCCESS)
      return 0;
    std::stringstream ss;
    ss << where << " failed with psz error code " << ec << " (" << psz_error_string(ec) << ")";
    return set_error(20, ss.str());
  }

#define lp_check_cuda_error(call)                                                                  \
  {                                                                                                \
    cudaError_t ec = (call);                                                                       \
    if (ec != cudaSuccess)                                                                         \
      return cuda_error(ec);                                                                       \
  }

  // states and context
  struct backend_state {
    cudaStream_t stream = nullptr;

    psz_ctx *enc = nullptr;
    psz_dtype enc_dtype = F4;
    psz_len enc_len = {0, 0, 0};
    bool enc_3stage = false;
    bool enc_dirty = false;
    void *d_in = nullptr;
    size_t d_in_cap = 0;

    psz_ctx *dec = nullptr;
    psz_header dec_header;
    void *d_archive = nullptr;
    size_t d_archive_cap = 0;
    void *d_out = nullptr;
    size_t d_out_cap = 0;

    ~backend_state() {
      if (enc)
        psz_free(enc);
      if (dec)
        psz_free(dec);
      if (d_in)
        cudaFree(d_in);
      if (d_archive)
        cudaFree(d_archive);
      if (d_out)
        cudaFree(d_out);
      if (stream)
        cudaStreamDestroy(stream);
    }
  };

  int ensure_stream() {
    if (not state)
      state = std::make_shared<backend_state>();
    if (not state->stream)
      lp_check_cuda_error(cudaStreamCreate(&state->stream));
    return 0;
  }

  int ensure_device_buffer(void *&ptr, size_t &cap, size_t bytes) {
    if (cap >= bytes)
      return 0;
    if (ptr)
      lp_check_cuda_error(cudaFree(ptr));
    ptr = nullptr;
    cap = 0;
    lp_check_cuda_error(cudaMalloc(&ptr, bytes));
    cap = bytes;
    return 0;
  }

  int ensure_encoder(psz_dtype dtype, psz_len len, bool three_stage) {
    auto &s = *state;
    bool const same = s.enc and s.enc_dtype == dtype and s.enc_len.x == len.x and
                      s.enc_len.y == len.y and s.enc_len.z == len.z and s.enc_3stage == three_stage;
    if (same)
      return 0;
    if (s.enc)
      psz_free(s.enc);
    s.enc = three_stage ? psz_compress_init_3stage(dtype, len, s.stream)
                        : psz_compress_init(dtype, len, s.stream);
    if (not s.enc)
      return psz_error(psz_last_error(), "psz_compress_init");
    s.enc_dtype = dtype;
    s.enc_len = len;
    s.enc_3stage = three_stage;
    s.enc_dirty = false;
    return 0;
  }

  template <class T> int compress_typed(pressio_data const *input, pressio_data *output) {
    if (int rc = ensure_stream(); rc != 0)
      return rc;
    auto &s = *state;

    std::vector<size_t> dims = input->normalized_dims(3, 1);
    psz_len len = {dims.at(0), dims.at(1), dims.at(2)};

    psz_ppl pipeline{to_psz_predictor(predictor_type), to_psz_hist(histogram_type),
                     to_psz_codec1(codec1_type), to_psz_codec2(codec2_type)};

    if (int rc = ensure_encoder(to_psz_dtype(input->dtype()), len, pipeline.codec2 != CodecNull);
        rc != 0)
      return rc;
    if (int rc = ensure_device_buffer(s.d_in, s.d_in_cap, input->size_in_bytes()); rc != 0)
      return rc;

    T *d_in = static_cast<T *>(s.d_in);
    lp_check_cuda_error(cudaMemcpyAsync(d_in, input->data(), input->size_in_bytes(),
                                        cudaMemcpyHostToDevice, s.stream));

    double eb = err_bnd;
    int ec = PSZ_SUCCESS;
    if (eb_mode == "rel") {
      if constexpr (std::is_same<T, float>::value)
        eb *= psz_compress_extrema_float(s.enc, d_in).rng;
      else
        eb *= psz_compress_extrema_double(s.enc, d_in).rng;
      ec = psz_last_error();
    }

    // Inline reset if cusz:reset was not issued.
    if (ec == PSZ_SUCCESS and s.enc_dirty)
      ec = psz_compress_reset(s.enc);

    s.enc_dirty = true;
    if (ec == PSZ_SUCCESS) {
      if constexpr (std::is_same<T, float>::value)
        ec = psz_compress_process_float(s.enc, pipeline, eb, d_in);
      else
        ec = psz_compress_process_double(s.enc, pipeline, eb, d_in);
    }

    psz_header header;
    uint8_t *d_archive = nullptr;
    size_t archive_bytes = 0;
    if (ec == PSZ_SUCCESS)
      ec = psz_compress_archive(s.enc, &header, &d_archive, &archive_bytes);
    if (ec != PSZ_SUCCESS)
      return psz_error(ec, "psz_compress");

    // The device archive already begins with its psz_header.
    if (not output->has_data() or output->size_in_bytes() != archive_bytes)
      *output = pressio_data::owning(pressio_byte_dtype, {archive_bytes});
    lp_check_cuda_error(cudaMemcpyAsync(output->data(), d_archive, archive_bytes,
                                        cudaMemcpyDeviceToHost, s.stream));
    lp_check_cuda_error(cudaStreamSynchronize(s.stream));
    return 0;
  }

  template <class T> int decompress_typed(pressio_data const *input, pressio_data *output) {
    size_t const archive_bytes = input->size_in_bytes();
    if (archive_bytes < sizeof(psz_header))
      return set_error(21, "input too small for psz header");

    psz_header header;
    memcpy(&header, input->data(), sizeof(psz_header));
    if (pszheader_filesize(&header) != archive_bytes)
      return set_error(22, "input is not a psz archive: header size does not match input size");
    if (pszheader_uncompressed_len(&header) * sizeof(T) != output->size_in_bytes())
      return set_error(23, "output size does not match the archive");

    if (int rc = ensure_stream(); rc != 0)
      return rc;
    auto &s = *state;

    if (int rc = ensure_device_buffer(s.d_archive, s.d_archive_cap, archive_bytes); rc != 0)
      return rc;
    if (int rc = ensure_device_buffer(s.d_out, s.d_out_cap, output->size_in_bytes()); rc != 0)
      return rc;

    // CAVEAT: The ctx caches the header: reuse only on an identical one.
    if (not s.dec or memcmp(&header, &s.dec_header, sizeof(psz_header)) != 0) {
      if (s.dec)
        psz_free(s.dec);
      s.dec = psz_decompress_init(&header, s.stream);
      if (not s.dec)
        return psz_error(psz_last_error(), "psz_decompress_init");
      memcpy(&s.dec_header, &header, sizeof(psz_header));
    }

    lp_check_cuda_error(cudaMemcpyAsync(s.d_archive, input->data(), archive_bytes,
                                        cudaMemcpyHostToDevice, s.stream));

    auto *d_archive = static_cast<uint8_t *>(s.d_archive);
    int ec;
    if constexpr (std::is_same<T, float>::value)
      ec = psz_decompress_process_float(s.dec, d_archive, archive_bytes,
                                        static_cast<float *>(s.d_out));
    else
      ec = psz_decompress_process_double(s.dec, d_archive, archive_bytes,
                                         static_cast<double *>(s.d_out));
    if (ec != PSZ_SUCCESS)
      return psz_error(ec, "psz_decompress");

    lp_check_cuda_error(cudaMemcpyAsync(output->data(), s.d_out, output->size_in_bytes(),
                                        cudaMemcpyDeviceToHost, s.stream));
    lp_check_cuda_error(cudaStreamSynchronize(s.stream));
    return 0;
  }

  int major_version() const override { return CUSZ_MAJOR_VERSION; }
  int minor_version() const override { return CUSZ_MINOR_VERSION; }
  int patch_version() const override { return CUSZ_PATCH_VERSION; }
  const char *version() const override { return "0.19"; }
  const char *prefix() const override { return "cusz"; }

  pressio_options get_metrics_results_impl() const override {
    pressio_options metrics;
    return metrics;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<cusz_compressor_plugin>(*this);
  }

  double err_bnd = 1e-5;
  std::string eb_mode = "abs";
  std::string histogram_type = "generic";
  std::string predictor_type = "lrz";
  std::string codec1_type = "hfr-pbkc";
  std::string codec2_type = "none";

private:
  std::shared_ptr<backend_state> state;
};

pressio_register registration(compressor_plugins(), "cusz",
                              []() { return compat::make_unique<cusz_compressor_plugin>(); });

} // namespace cusz_ns
} // namespace compressors
} // namespace libpressio
