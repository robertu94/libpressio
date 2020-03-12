#include "pressio_metrics.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"

extern "C" {

void pressio_metrics_free(struct pressio_metrics* metrics) {
  delete metrics;
}

struct pressio_options* pressio_metrics_get_results(struct pressio_metrics const* metrics) {
  return new pressio_options((*metrics)->get_metrics_results());
}

struct pressio_options* pressio_metrics_get_options(struct pressio_metrics const* metrics) {
  return new pressio_options((*metrics)->get_options());
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


}
