#include <libdistributed_task_manager.h>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/compat/memory.h"
#include "libdistributed/libdistributed_work_queue.h"
#include "libpressio_ext/cpp/serializable.h"
#include "libpressio_ext/cpp/distributed_manager.h"

class many_dependent_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "many_dependent:compressor", compressor_id, compressor);
    options.copy_from(manager.get_options());
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int>(pressio_thread_safety_multiple));
    options.copy_from(manager.get_configuration());
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "many_dependent:compressor", compressor_plugins(), compressor_id, compressor);
    manager.set_options(options);
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
    using request_t = std::tuple<int, pressio_options>;
    using response_t = std::tuple<int, pressio_options, pressio_data>;
    using distributed::queue::TaskManager;
    std::vector<request_t> requests;
    requests.emplace_back(
      0,
      pressio_options{}
    );
    size_t outstanding = 1;
    size_t next_task = 1;


    manager.work_queue(
        std::begin(requests), std::end(requests),
        [&inputs, &outputs, this](request_t const& request) {
          auto index = std::get<0>(request);
          compressor->compress(inputs[index], outputs[index]);
          pressio_options options = compressor->get_metrics_results();

          pressio_options new_options;
          for (size_t i = 0; i < to_names.size(); ++i) {
            new_options.set(from_names[i], options.get(from_names[i]));
          }


          return response_t{index, std::move(new_options), std::move(*outputs[index])};
        },
        [&outputs, &outstanding, &next_task, this](response_t response, TaskManager<request_t, MPI_Comm>& task_manager) {
          //one less outstanding
          outstanding--;

          auto index = std::get<0>(response);
          *outputs[index] = std::move(std::get<2>(response));

          //determine the next set of options
          auto options = std::move(std::get<1>(response));
          pressio_options new_options;
          for (size_t i = 0; i < to_names.size(); ++i) {
            new_options.set(to_names[i], options.get(from_names[i]));
          }


          //push search_requests to fill workers
          while(outstanding < task_manager.num_workers() && next_task < outputs.size()) {
            request_t request{next_task, new_options};
            task_manager.push(request);
            outstanding++;
            next_task++;
          }

        }
        );
    return 0;
  }

  int decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data* >& outputs) override {
    using request_t = std::tuple<int>;
    using response_t = std::tuple<int, pressio_data>;
    std::vector<request_t> requests(inputs.size());
    std::iota(std::begin(requests), std::end(requests), 0);

    manager.work_queue(
        std::begin(requests), std::end(requests),
        [&outputs, &inputs, this](request_t request) {
          size_t index = std::get<0>(request);
          compressor->decompress(inputs[index], outputs[index]);
          return response_t{index, std::move(*outputs[index])};
        },
        [&outputs](response_t const& response) {
          size_t index = std::get<0>(response);
          *outputs[index] = std::get<1>(response);
        });
    return 0;
  }


  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  const char* version() const override { return "0.0.1"; }

  const char* prefix() const override { return "many_dependent"; }

  void set_name_impl(std::string const& name) override {
    compressor->set_name(name + "/" + compressor->prefix());
    manager.set_name(name);
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<many_dependent_compressor_plugin>(*this);
  }

private:
  std::vector<std::string> from_names;
  std::vector<std::string> to_names;

  pressio_distributed_manager manager = pressio_distributed_manager(
      /*max_ranks_per_worker*/pressio_distributed_manager::unlimited,
      /*max_masters*/1
      );

  pressio_compressor compressor = compressor_plugins().build("noop");
  std::string compressor_id = "noop";
};

static pressio_register
    compressor_many_timesteps_plugin(compressor_plugins(), "many_dependent", []() {
      return compat::make_unique<many_dependent_compressor_plugin>();
    });
