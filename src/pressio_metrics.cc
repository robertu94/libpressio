#include "pressio_metrics.h"
#include "libpressio_ext/cpp/metrics.h"

extern "C" {

void pressio_metrics_free(struct pressio_metrics* metrics) {
  delete metrics;
}

struct pressio_options* pressio_metrics_get_results(struct pressio_metrics const* metrics) {
  return (*metrics)->get_metrics_results();
}

}
