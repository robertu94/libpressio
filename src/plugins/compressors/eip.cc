#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cusz/cusz_version.h>
#ifndef EIP_MAJOR_VERSION
#define EIP_MAJOR_VERSION CUSZ_MAJOR_VERSION
#define EIP_MINOR_VERSION CUSZ_MINOR_VERSION
#define EIP_PATCH_VERSION CUSZ_PATCH_VERSION
#endif
#if __has_include(<eip.h>)
#include <eip.h>
#elif __has_include(<cusz/include/eip.h>)
#include <cusz/include/eip.h>
#else
#error "EIP header eip.h not found"
#endif

namespace libpressio {
namespace compressors {
namespace eip_ns {

class eip_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override {
    struct pressio_options options;
    set(options, "pressio:abs", err_bnd);
    set(options, "eip:preset", preset);
    set(options, "eip:tile", tile);
    set(options, "eip:decode", decode_mode);
    set_type(options, "eip:reset", pressio_option_int32_type);
    return options;
  }

  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_serialized);
    set(options, "pressio:stability", "experimental");
    set(options, "eip:preset", std::vector<std::string>{"a", "b"});
    set(options, "eip:tile", std::vector<std::string>{"1Ki", "2Ki"});
    set(options, "eip:decode", std::vector<std::string>{"cpu", "gpu"});
    return options;
  }

  struct pressio_options get_documentation_impl() const override {
    struct pressio_options options;
    set(options, "pressio:description", R"(EIP: encode-in-place add-on of cuSZ, float only)");
    set(options, "pressio:abs", "absolute error bound");
    set(options, "eip:preset", "encoder preset: a (v3+v1) or b (v7, default)");
    set(options, "eip:tile", "3D tile size: 1Ki or 2Ki; applies on the next compress");
    set(options, "eip:decode", "decompress path: cpu or gpu");
    set(options, "eip:reset",
        "write-only trigger: set to 1 to clear the reused encode cursor between runs");
    return options;
  }

  int set_options_impl(struct pressio_options const &options) override {
    get(options, "pressio:abs", &err_bnd);
    get(options, "eip:preset", &preset);
    get(options, "eip:tile", &tile);
    get(options, "eip:decode", &decode_mode);

    if (preset != "a" and preset != "b")
      return set_error(3, "eip:preset must be a or b");
    if (tile != "1Ki" and tile != "2Ki")
      return set_error(3, "eip:tile must be 1Ki or 2Ki");
    if (decode_mode != "cpu" and decode_mode != "gpu")
      return set_error(3, "eip:decode must be cpu or gpu");

    int32_t do_reset = 0;
    if (get(options, "eip:reset", &do_reset) == pressio_options_key_set and do_reset != 0)
      return reset();

    return 0;
  }

  // clear between runs
  int reset() {
    if (not state or not state->enc or not state->enc_dirty)
      return 0;
    if (int ec = eip_compress_reset(state->enc); ec != EIP_SUCCESS)
      return eip_error(ec, "eip_compress_reset");
    state->enc_dirty = false;
    return 0;
  }

  int cuda_error(cudaError_t ec) { return set_error(12, cudaGetErrorString(ec)); }

  int eip_error(int ec, char const *where) {
    if (ec == EIP_SUCCESS)
      return 0;
    char const *name = "unknown";
    switch (ec) {
    case EIP_INVALID_INPUT:
      name = "invalid input";
      break;
    case EIP_COMPRESS_FAILED:
      name = "compress failed";
      break;
    case EIP_DECOMPRESS_FAILED:
      name = "decompress failed";
      break;
    case EIP_BAD_ARCHIVE:
      name = "bad archive";
      break;
    }
    std::stringstream ss;
    ss << where << " failed with eip error code " << ec << " (" << name << ")";
    return set_error(20, ss.str());
  }

#define lp_eip_check_cuda(call)                                                                    \
  {                                                                                                \
    cudaError_t ec = (call);                                                                       \
    if (ec != cudaSuccess)                                                                         \
      return cuda_error(ec);                                                                       \
  }

  // Device state, shared across clone()s.
  struct backend_state {
    cudaStream_t stream = nullptr;

    eip_ctx *enc = nullptr;
    size_t enc_x = 0, enc_y = 0, enc_z = 0;
    eip_tile enc_tile = EIP_TILE_1Ki;
    bool enc_dirty = false;
    void *d_in = nullptr;
    size_t d_in_cap = 0;

    eip_ctx *dec = nullptr;
    eip_header dec_header;
    void *d_archive = nullptr;
    size_t d_archive_cap = 0;
    void *d_out = nullptr;
    size_t d_out_cap = 0;

    ~backend_state() {
      if (enc)
        eip_free(enc);
      if (dec)
        eip_free(dec);
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
      lp_eip_check_cuda(cudaStreamCreate(&state->stream));
    return 0;
  }

  int ensure_device_buffer(void *&ptr, size_t &cap, size_t bytes) {
    if (cap >= bytes)
      return 0;
    if (ptr)
      lp_eip_check_cuda(cudaFree(ptr));
    ptr = nullptr;
    cap = 0;
    lp_eip_check_cuda(cudaMalloc(&ptr, bytes));
    cap = bytes;
    return 0;
  }

  int ensure_encoder(size_t x, size_t y, size_t z, eip_tile tile_variant) {
    auto &s = *state;
    bool const same =
        s.enc and s.enc_x == x and s.enc_y == y and s.enc_z == z and s.enc_tile == tile_variant;
    if (same)
      return 0;
    if (s.enc)
      eip_free(s.enc);
    s.enc = eip_compress_init_float(static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                                    static_cast<uint32_t>(z), tile_variant, s.stream);
    if (not s.enc)
      return eip_error(eip_last_error(), "eip_compress_init_float");
    s.enc_x = x;
    s.enc_y = y;
    s.enc_z = z;
    s.enc_tile = tile_variant;
    s.enc_dirty = true; // first compress resets
    return 0;
  }

  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    if (input->dtype() != pressio_float_dtype)
      return set_error(1, "EIP supports float only");

    if (int rc = ensure_stream(); rc != 0)
      return rc;
    auto &s = *state;

    std::vector<size_t> dims = input->normalized_dims(3, 1);
    eip_tile const tile_variant = tile == "2Ki" ? EIP_TILE_2Ki : EIP_TILE_1Ki;
    if (int rc = ensure_encoder(dims.at(0), dims.at(1), dims.at(2), tile_variant); rc != 0)
      return rc;
    if (int rc = ensure_device_buffer(s.d_in, s.d_in_cap, input->size_in_bytes()); rc != 0)
      return rc;

    lp_eip_check_cuda(cudaMemcpyAsync(s.d_in, input->data(), input->size_in_bytes(),
                                      cudaMemcpyHostToDevice, s.stream));

    if (int ec = eip_set_preset(s.enc, preset[0]); ec != EIP_SUCCESS)
      return eip_error(ec, "eip_set_preset");

    if (s.enc_dirty) {
      if (int ec = eip_compress_reset(s.enc); ec != EIP_SUCCESS)
        return eip_error(ec, "eip_compress_reset");
    }
    s.enc_dirty = true;

    eip_header header;
    void *d_archive = nullptr;
    size_t archive_bytes = 0;

    int ec = eip_compress_process_float(s.enc, err_bnd, static_cast<float *>(s.d_in),
                                        input->num_elements(), std::numeric_limits<size_t>::max(),
                                        &header, &d_archive, &archive_bytes);
    if (ec != EIP_SUCCESS)
      return eip_error(ec, "eip_compress_process_float");

    if (not output->has_data() or output->size_in_bytes() != archive_bytes)
      *output = pressio_data::owning(pressio_byte_dtype, {archive_bytes});
    lp_eip_check_cuda(cudaMemcpyAsync(output->data(), d_archive, archive_bytes,
                                      cudaMemcpyDeviceToHost, s.stream));
    lp_eip_check_cuda(cudaStreamSynchronize(s.stream));
    return 0;
  }

  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    size_t const archive_bytes = input->size_in_bytes();
    if (archive_bytes < sizeof(eip_header))
      return set_error(21, "input too small for eip header");

    eip_header header;
    memcpy(&header, input->data(), sizeof(eip_header));
    if (eipheader_compressed_bytes(&header) != archive_bytes)
      return set_error(22, "input is not an eip archive: header size does not match input size");

    size_t const out_bytes = eipheader_uncompressed_bytes(&header);
    if (not output->has_data() or output->size_in_bytes() != out_bytes)
      *output = pressio_data::owning(pressio_float_dtype, {eipheader_len_linear(&header)});

    if (int rc = ensure_stream(); rc != 0)
      return rc;
    auto &s = *state;

    // The ctx caches the header: reuse only on an identical one.
    if (not s.dec or memcmp(&header, &s.dec_header, sizeof(eip_header)) != 0) {
      if (s.dec)
        eip_free(s.dec);
      s.dec = eip_decompress_init_float(&header, s.stream);
      if (not s.dec)
        return eip_error(eip_last_error(), "eip_decompress_init_float");
      memcpy(&s.dec_header, &header, sizeof(eip_header));
    }

    if (decode_mode == "cpu") {
      int ec = eip_decompress_process_float_cpu(s.dec, input->data(), archive_bytes,
                                                static_cast<float *>(output->data()));
      return eip_error(ec, "eip_decompress_process_float_cpu");
    }

    if (int rc = ensure_device_buffer(s.d_archive, s.d_archive_cap, archive_bytes); rc != 0)
      return rc;
    if (int rc = ensure_device_buffer(s.d_out, s.d_out_cap, out_bytes); rc != 0)
      return rc;

    lp_eip_check_cuda(cudaMemcpyAsync(s.d_archive, input->data(), archive_bytes,
                                      cudaMemcpyHostToDevice, s.stream));
    if (int ec = eip_decompress_process_float(s.dec, s.d_archive, archive_bytes,
                                              static_cast<float *>(s.d_out));
        ec != EIP_SUCCESS)
      return eip_error(ec, "eip_decompress_process_float");
    lp_eip_check_cuda(
        cudaMemcpyAsync(output->data(), s.d_out, out_bytes, cudaMemcpyDeviceToHost, s.stream));
    lp_eip_check_cuda(cudaStreamSynchronize(s.stream));
    return 0;
  }

  int major_version() const override { return EIP_MAJOR_VERSION; }
  int minor_version() const override { return EIP_MINOR_VERSION; }
  int patch_version() const override { return EIP_PATCH_VERSION; }
  const char *version() const override { return "0.19-sync"; }
  const char *prefix() const override { return "eip"; }

  pressio_options get_metrics_results_impl() const override { return pressio_options(); }

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<eip_compressor_plugin>(*this);
  }

  double err_bnd = 1e-3;
  std::string preset = "b";
  std::string tile = "1Ki";
  std::string decode_mode = "cpu";

private:
  std::shared_ptr<backend_state> state;
};

pressio_register registration(compressor_plugins(), "eip",
                              []() { return compat::make_unique<eip_compressor_plugin>(); });

} // namespace eip_ns
} // namespace compressors
} // namespace libpressio
