#include <libdistributed_task_manager.h>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/subgroup_manager.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"
#include "libdistributed/libdistributed_work_queue.h"
#include "libpressio_ext/cpp/serializable.h"
#include "libpressio_ext/cpp/distributed_manager.h"
#include <mpi.h>

namespace libpressio { namespace many_dependent_compressor {

class many_dependent_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "many_dependent:compressor", compressor_id, compressor);
    options.copy_from(manager.get_options());
    options.copy_from(subgroups.get_options());
    set(options, "many_dependent:to_names", to_names);
    set(options, "many_dependent:from_names", from_names);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    options.copy_from(manager.get_configuration());
    options.copy_from(subgroups.get_configuration());
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(Uses MPI to compress multiple buffers in parallel using results of previously
      successful compressions to guide future compressions)");
    set_meta_docs(options, "many_dependent:compressor", "the name of the compressor to pipeline over using MPI", compressor);
    set(options, "many_dependent:to_names", "list of options to set on each launch");
    set(options, "many_dependent:from_names", "list of metrics to pull the next set of configurations from");
    options.copy_from(manager.get_documentation());
    options.copy_from(subgroups.get_documentation());
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "many_dependent:compressor", compressor_plugins(), compressor_id, compressor);
    manager.set_options(options);
    subgroups.set_options(options);
    get(options, "many_dependent:to_names", &to_names);
    get(options, "many_dependent:from_names", &from_names);
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
    using request_t = std::tuple<int, pressio_options>; //index, metrics
    using response_t = std::tuple<int, pressio_options, std::vector<pressio_data>, int, std::string>; //index, metrics, compressed, error code, error_message
    using distributed::queue::TaskManager;
    std::vector<request_t> requests;
    requests.emplace_back(
      0,
      pressio_options{}
    );
    size_t outstanding = 1;
    size_t next_task = 1;

    if(subgroups.normalize_and_validate(inputs, outputs)) {
      return set_error(subgroups.error_code(), subgroups.error_msg());
    }


    int ret = 0;
    ret = manager.work_queue(
        std::begin(requests), std::end(requests),
        [&inputs, &outputs, this](request_t request, distributed::queue::TaskManager<request_t, MPI_Comm>& task_manager) {

          std::vector<pressio_data> output_data;
          auto index = std::get<0>(request);
          auto request_options = std::move(std::get<1>(request));
          request_options.set(compressor->get_name(), "distributed:mpi_comm", (void*)task_manager.get_subcommunicator());

          if(compressor->set_options(request_options)) {
            return response_t{index, pressio_options{}, output_data, compressor->error_code(), compressor->error_msg()};
          }
          auto input_data_ptrs = subgroups.get_input_group(inputs, index);
          auto output_data_ptrs = subgroups.get_output_group(outputs, index);
          if(compressor->compress_many(
                input_data_ptrs.data(),
                input_data_ptrs.data() + input_data_ptrs.size(),
                output_data_ptrs.data(),
                output_data_ptrs.data() + output_data_ptrs.size()
                )) {
            for (auto& i : output_data_ptrs) {
              output_data.emplace_back(std::move(*i));
            }
            
            return response_t{index, pressio_options{}, output_data, compressor->error_code(), compressor->error_msg()};
          }

          for (auto& i : output_data_ptrs) {
            output_data.emplace_back(std::move(*i));
          }

          pressio_options metrics_results = compressor->get_metrics_results();
          int error_code = compressor->error_code();
          std::string error_msg = compressor->error_msg();

          pressio_options new_options;
          for (size_t i = 0; i < to_names.size(); ++i) {
            auto option_it = metrics_results.find(from_names[i]);
            if(option_it != metrics_results.end()){
              new_options.set(from_names[i], option_it->second);
            } else {
              error_code = 3;
              error_msg = std::string("invalid option in from_names: ") + from_names[i];
              break;
            }
          }


          return response_t{index, std::move(new_options), output_data, error_code, error_msg};
        },
        [&outputs, &outstanding, &next_task, &ret, this](response_t response, TaskManager<request_t, MPI_Comm>& task_manager) {
          //one less outstanding
          outstanding--;

          auto index = std::get<0>(response);
          auto ec = std::get<3>(response);
          if(ec) {
            ret |= set_error(ec, std::get<4>(response));
          }

          
          //retrieve outputs
          auto output_data = std::move(std::get<2>(response));
          size_t out_idx=0;
          for (size_t i =0; i < subgroups.effective_output_groups().size(); ++i) {
            if(subgroups.effective_output_groups()[i] == index) {
              *outputs[i] = std::move(output_data[out_idx++]);
            }
          }

          //determine the next set of options
          auto options = std::move(std::get<1>(response));
          pressio_options new_options;
          for (size_t i = 0; i < to_names.size(); ++i) {
            auto option_it = options.find(from_names[i]);
            if(option_it != options.end()){
              new_options.set(to_names[i], option_it->second);
            } else {
              set_error(3, "invalid option in from_names" + from_names[i]);
              break;
            }
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
    return ret;
  }

  int decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data* >& outputs) override {
    using request_t = std::tuple<int>;
    using response_t = std::tuple<int, pressio_data, int, std::string>;
    std::vector<request_t> requests(inputs.size());
    std::iota(std::begin(requests), std::end(requests), 0);
    int ret = 0;


    manager.work_queue(
        std::begin(requests), std::end(requests),
        [&outputs, &inputs, this](request_t request) {
          size_t index = std::get<0>(request);
          compressor->decompress(inputs[index], outputs[index]);
          return response_t{index, std::move(*outputs[index]), error_code(), error_msg()};
        },
        [&outputs,&ret, this](response_t const& response) {
          size_t index = std::get<0>(response);
          size_t ec = std::get<2>(response);
          if(ec) {
            ret |= set_error(ec, std::get<3>(response));
          }
          *outputs[index] = std::get<1>(response);
        });
    return ret;
  }


  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  const char* version() const override { return "0.0.1"; }

  const char* prefix() const override { return "many_dependent"; }

  void set_name_impl(std::string const& name) override {
    compressor->set_name(name + "/" + compressor->prefix());
    manager.set_name(name);
    subgroups.set_name(name);
  }

  pressio_options get_metrics_results_impl() const override {
    return compressor->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<many_dependent_compressor_plugin>(*this);
  }

private:
  std::vector<std::string> from_names;
  std::vector<std::string> to_names;

  pressio_subgroup_manager subgroups;
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
} }
