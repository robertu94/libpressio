#include "libdistributed/libdistributed_work_queue.h"
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_compressor.h"
#include "pressio_data.h"
#include "pressio_options.h"
#include <cstddef>
#include <libdistributed_task_manager.h>
#include <libdistributed_work_queue_options.h>
#include <libpressio_ext/cpp/distributed_manager.h>
#include <libpressio_ext/cpp/serializable.h>
#include <libpressio_ext/cpp/subgroup_manager.h>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <mpi.h>

namespace libpressio { namespace  many_independent {
  

class many_independent_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "many_independent:compressor", compressor_id, compressor);
    set(options, "many_independent:bcast_outputs", bcast_outputs);
    options.copy_from(manager.get_options());
    options.copy_from(subgroups.get_options());
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "many_independent:compressor", "compressor to parallelize using MPI", compressor);
    set(options, "many_independent:bcast_outputs", "true if all ranks have the same outputs otherwise just the root has the outputs");
    set(options, "pressio:description", R"(Uses MPI to compress multiple buffers in parallel)");
    options.copy_from(manager.get_documentation());
    options.copy_from(subgroups.get_documentation());
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    options.copy_from(compressor->get_configuration());
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    options.copy_from(manager.get_configuration());
    options.copy_from(subgroups.get_configuration());
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    pressio_data tmp;

    get_meta(options, "many_independent:compressor", compressor_plugins(), compressor_id, compressor);
    get(options, "many_independent:bcast_outputs", &bcast_outputs);
    manager.set_options(options);
    subgroups.set_options(options);
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
    return common_many_impl(inputs, outputs, [this](
          pressio_data const** inputs_begin,
          pressio_data const** inputs_end,
          pressio_data ** outputs_begin,
          pressio_data ** outputs_end
          ){ 
          return compressor->compress_many(inputs_begin, inputs_end, outputs_begin, outputs_end);
        });
  }

  int decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data* >& outputs) override {
    return common_many_impl(inputs, outputs, [this](
          pressio_data const** inputs_begin,
          pressio_data const** inputs_end,
          pressio_data** outputs_begin,
          pressio_data** outputs_end
          ){ 
          return compressor->decompress_many(inputs_begin, inputs_end, outputs_begin, outputs_end);
        });
  }


  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  const char* version() const override { return "0.0.1"; }

  const char* prefix() const override { return "many_independent"; }

  void set_name_impl(std::string const& name) override {
    if(name != "") {
      compressor->set_name(name + "/" + compressor->prefix());
    } else {
      compressor->set_name(name);
    }
    manager.set_name(name);
    subgroups.set_name(name);
  }

  pressio_options get_metrics_results_impl() const override {
    return compressor->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<many_independent_compressor_plugin>(*this);
  }

private:
  template <class Action>
  int common_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs, Action&& action)
  {
    using request_t = std::tuple<int>; //group_idx
    using response_t = std::tuple<int, int, std::vector<pressio_data>, std::string>; //group_idx, status, data, err_msg

    if(subgroups.normalize_and_validate(inputs, outputs)) {
      return set_error(subgroups.error_code(), subgroups.error_msg());
    }

    auto indicies = std::set<request_t>(std::begin(subgroups.effective_input_groups()),
    std::end(subgroups.effective_input_groups()));

    int status = 0;
    status = manager.work_queue(
        std::begin(indicies), std::end(indicies),
        [this, &inputs, &outputs, &action](request_t request, distributed::queue::TaskManager<request_t, MPI_Comm>& task_manager) {
          //setup the work groups
          int idx = std::get<0>(request);
          auto input_data = subgroups.get_input_group(inputs, idx);
          auto output_data_ptrs = subgroups.get_output_group(outputs, idx);

          pressio_options sub_options;
          sub_options.set(compressor->get_name(),
              "distributed:mpi_comm",
              userdata(
                (void*)new MPI_Comm(*task_manager.get_subcommunicator()),
                nullptr,
                newdelete_deleter<MPI_Comm>(),
                newdelete_copy<MPI_Comm>()
                )
            );
          compressor->set_options(sub_options);

          //run the action: either compression or decompression
          int status = action(
              input_data.data(),
              input_data.data() + input_data.size(),
              output_data_ptrs.data(),
              output_data_ptrs.data() + output_data_ptrs.size()
              );


          //move the compressed buffers to the response to be transferred
          std::vector<pressio_data> output_data;
          output_data.reserve(output_data_ptrs.size());
          for (auto output_data_ptr : output_data_ptrs) {
            output_data.emplace_back(std::move(*output_data_ptr));
          }


          return response_t{idx, status, std::move(output_data), compressor->error_msg()};
        },
        [&outputs,&status, this](response_t response) {
          //retrive data and errors
          int idx = std::get<0>(response);
          status |= std::get<1>(response);
          if(std::get<1>(response)) {
            set_error(std::get<1>(response), std::get<3>(response));
          }

          //store the output into the appropriate buffers
          auto output_data = std::move(std::get<2>(response));
          size_t out_idx=0;
          for (size_t i =0; i < subgroups.effective_output_groups().size(); ++i) {
            if(subgroups.effective_output_groups()[i] == idx) {
              *outputs[i] = std::move(output_data[out_idx++]);
            }
          }
        });
    if(status == 0 && bcast_outputs) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        manager.bcast(outputs[i], 0);
      }
    }
    return status;
  }

  pressio_subgroup_manager subgroups;
  pressio_distributed_manager manager = pressio_distributed_manager(
      /*max_ranks_per_worker*/pressio_distributed_manager::unlimited,
      /*max_masters*/1
      );
  pressio_compressor compressor = compressor_plugins().build("noop");
  std::string compressor_id = "noop";
  int bcast_outputs = 1;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "many_independent", []() {
  return compat::make_unique<many_independent_compressor_plugin>();
});

} }
