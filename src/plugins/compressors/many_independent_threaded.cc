#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_compressor.h"
#include "pressio_data.h"
#include "pressio_options.h"
#include <cstddef>
#include <libpressio_ext/cpp/subgroup_manager.h>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <mpi.h>

namespace libpressio { namespace many_independent_threaded_ns {
  enum class MetricsAction {
    Ignore, /// do not collect metrics on sub operations at all
    Archive, /// save off the metrics object after compression
    Restore ///  restore metrics object during decompression
  };

class many_independent_threaded_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "many_independent_threaded:compressor", compressor_id, compressor);
    options.copy_from(subgroups.get_options());
    set(options, "many_independent_threaded:nthreads", nthreads);
    set(options, "many_independent_threaded:collect_metrics_on_compression", collect_metrics_on_compression);
    set(options, "many_independent_threaded:collect_metrics_on_decompression", collect_metrics_on_decompression);
    set(options, "many_independent_threaded:preserve_metrics", preserve_metrics);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    options.copy_from(subgroups.get_configuration());
    options.copy_from(compressor->get_configuration());
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "many_independent_threaded:compressor", "the child compressor to use", compressor);
    options.copy_from(subgroups.get_documentation());
    set(options, "pressio:description", R"(Uses OpenMP to compress multiple buffers in parallel)");
    set(options, "many_independent_threaded:nthreads", R"(number of threads to use for compression)");
    set(options, "many_independent_threaded:collect_metrics_on_compression", R"(collect metrics after compression)");
    set(options, "many_independent_threaded:collect_metrics_on_decompression", R"(collect metrics after decompression)");
    set(options, "many_independent_threaded:preserve_metrics", R"(preserve metrics after compression, and restore them before decompression)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    pressio_data tmp;

    get_meta(options, "many_independent_threaded:compressor", compressor_plugins(), compressor_id, compressor);
    subgroups.set_options(options);
    auto tmp_threads = nthreads;
    if (get(options, "many_independent_threaded:nthreads", &tmp_threads) == pressio_options_key_set) {
      if(tmp_threads >= 1) {
        nthreads = tmp_threads;
      } else {
        return set_error(1, "invalid thread count");
      }
    }
    get(options, "many_independent_threaded:collect_metrics_on_compression", &collect_metrics_on_compression);
    get(options, "many_independent_threaded:collect_metrics_on_decompression", &collect_metrics_on_decompression);
    get(options, "many_independent_threaded:preserve_metrics", &preserve_metrics);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    compat::span<const pressio_data*> inputs(&input, 1);
    compat::span<pressio_data*> outputs(&output, 1);
    return compress_many_impl(inputs, outputs);
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    compat::span<const pressio_data*> inputs(&input, 1);
    compat::span<pressio_data*> outputs(&output, 1);
    return decompress_many_impl(inputs, outputs);
  }

  int compress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs) override {
    MetricsAction archive_action = MetricsAction::Ignore;
    if(preserve_metrics) {
      archive_action = MetricsAction::Archive;
    }

    return common_many_impl(inputs, outputs, [](
          pressio_compressor& local_compressor,
          pressio_data const** inputs_begin,
          pressio_data const** inputs_end,
          pressio_data ** outputs_begin,
          pressio_data ** outputs_end
          ){ 
          return local_compressor->compress_many(inputs_begin, inputs_end, outputs_begin, outputs_end);
        }, collect_metrics_on_compression, archive_action);
  }

  int decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data* >& outputs) override {
    MetricsAction restore_action = MetricsAction::Ignore;
    if(preserve_metrics) {
      restore_action = MetricsAction::Restore;
    }
    return common_many_impl(inputs, outputs, [](
          pressio_compressor& local_compressor,
          pressio_data const** inputs_begin,
          pressio_data const** inputs_end,
          pressio_data** outputs_begin,
          pressio_data** outputs_end
          ){ 
          return local_compressor->decompress_many(inputs_begin, inputs_end, outputs_begin, outputs_end);
        }, collect_metrics_on_decompression, restore_action);
  }


  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  const char* version() const override { return "0.0.1"; }

  const char* prefix() const override { return "many_independent_threaded"; }

  void set_name_impl(std::string const& name) override {
    compressor->set_name(name + "/" + compressor->prefix());
    subgroups.set_name(name);
  }

  pressio_options get_metrics_results_impl() const override {
    return metrics_results;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<many_independent_threaded_compressor_plugin>(*this);
  }

private:
  template <class Action>
  int common_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs, Action&& action, bool collect_metrics, MetricsAction const metrics_action)
  {
    if(subgroups.normalize_and_validate(inputs, outputs)) {
      return set_error(subgroups.error_code(), subgroups.error_msg());
    }

    auto indicies = std::set<int>(std::begin(subgroups.effective_input_groups()), std::end(subgroups.effective_input_groups()));
    std::vector<int> indicies_vec(indicies.begin(), indicies.end());

    if(metrics_action == MetricsAction::Archive) {
      preserve_metrics_mem.resize(indicies_vec.size());
    }
 
    int status = 0;

    pressio_options tmp_metrics_results;

#pragma omp parallel default(none), shared(indicies_vec, inputs, outputs, action, status, collect_metrics, tmp_metrics_results, metrics_action), num_threads(nthreads)
#pragma omp for schedule(dynamic)
    for (size_t idx = 0; idx < indicies_vec.size(); ++idx) {
      auto input_data = subgroups.get_input_group(inputs, indicies_vec[idx]);
      auto output_data_ptrs = subgroups.get_output_group(outputs, indicies_vec[idx]);
      pressio_compressor thread_local_compressor;

      thread_local_compressor = compressor->clone();
      if(metrics_action == MetricsAction::Restore) {
        thread_local_compressor->set_metrics(std::move(preserve_metrics_mem.at(idx)));
      }

      //run the action: either compression or decompression
      int local_status = action(
          thread_local_compressor,
          input_data.data(),
          input_data.data() + input_data.size(),
          output_data_ptrs.data(),
          output_data_ptrs.data() + output_data_ptrs.size()
          );

      if(collect_metrics) {
#pragma omp critical
        {
          tmp_metrics_results.copy_from(thread_local_compressor->get_metrics_results());
        }
      }

      if(metrics_action == MetricsAction::Archive) {
        preserve_metrics_mem.at(idx) = std::move(*thread_local_compressor).get_metrics();
      }

      if(local_status) {
#pragma omp critical
        {
          set_error(compressor->error_code(), compressor->error_msg());
          status = local_status;
        }
#pragma omp cancel for
      }
    }

    if(collect_metrics) {
      metrics_results.copy_from(tmp_metrics_results);
    }

    return status;

  }

  pressio_subgroup_manager subgroups;
  pressio_options metrics_results;
  std::vector<pressio_metrics> preserve_metrics_mem;
  std::string compressor_id = "noop";
  pressio_compressor compressor = compressor_plugins().build("noop");
  uint32_t nthreads = 1;
  int32_t collect_metrics_on_decompression = 0;
  int32_t collect_metrics_on_compression = 0;
  int32_t preserve_metrics = 0;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "many_independent_threaded", []() {
  return compat::make_unique<many_independent_threaded_compressor_plugin>();
});

} }
