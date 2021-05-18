#include "pressio_metrics.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"

static compat::span<const pressio_data* const>
mk_span(struct pressio_data const* const* ptr, size_t size) {
  if(ptr == nullptr) return compat::span<const pressio_data* const>();
  else return compat::span<const pressio_data* const>(ptr, size);
}

extern "C" {

void pressio_metrics_free(struct pressio_metrics* metrics) {
  delete metrics;
}

struct pressio_options* pressio_metrics_get_results(struct pressio_metrics const* metrics) {
  return new pressio_options((*metrics)->get_metrics_results({}));
}

struct pressio_options* pressio_metrics_get_options(struct pressio_metrics const* metrics) {
  return new pressio_options((*metrics)->get_options());
}

struct pressio_options* pressio_metrics_get_configuration(struct pressio_metrics const* metrics) {
  return new pressio_options((*metrics)->get_configuration());
}

struct pressio_options* pressio_metrics_get_documentation(struct pressio_metrics const* metrics) {
  return new pressio_options((*metrics)->get_documentation());
}

int pressio_metrics_set_options(struct pressio_metrics const* metrics, struct pressio_options const* options){
  return (*metrics)->set_options(*options);
}

struct pressio_metrics* pressio_metrics_clone(struct pressio_metrics* metrics) {
  return new pressio_metrics((*metrics)->clone());
}

void pressio_metrics_set_name(struct pressio_metrics* metrics, const char* new_name) {
  (*metrics)->set_name(new_name);
}


const char* pressio_metrics_get_name(struct pressio_metrics const* metrics) {
  return (*metrics)->get_name().c_str();
}

struct pressio_options* pressio_metrics_evaluate(
    struct pressio_metrics* metrics,
    struct pressio_data const* uncompressed,
    struct pressio_data const* compressed,
    struct pressio_data const* decompressed
    ) {
  if(uncompressed) {
    (*metrics)->begin_compress(uncompressed, compressed);
    (*metrics)->end_compress(uncompressed, compressed, 0);
  }

  if(decompressed) {
    (*metrics)->begin_decompress(compressed, decompressed);
    (*metrics)->end_decompress(compressed, decompressed, 0);
  }

  return new pressio_options((*metrics)->get_metrics_results({}));
}

struct pressio_options* pressio_metrics_evaluate_many(
    struct pressio_metrics* metrics,
    struct pressio_data const* const* uncompressed, size_t n_uncompressed,
    struct pressio_data const* const* compressed, size_t n_compressed,
    struct pressio_data const* const* decompressed, size_t n_decompressed
    ) {
  auto uncompressed_s = mk_span(uncompressed, n_uncompressed);
  auto compressed_s = mk_span(compressed, n_compressed);
  auto decompressed_s = mk_span(decompressed, n_decompressed);

  if(uncompressed) {
    (*metrics)->begin_compress_many(uncompressed_s, compressed_s);
    (*metrics)->end_compress_many(uncompressed_s, compressed_s, 0);
  }

  if(decompressed) {
    (*metrics)->begin_decompress_many(compressed_s, decompressed_s);
    (*metrics)->end_decompress_many(compressed_s, decompressed_s, 0);
  }

  return new pressio_options((*metrics)->get_metrics_results({}));
}


}
