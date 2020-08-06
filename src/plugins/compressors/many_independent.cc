#include <cstddef>
#include <libdistributed_task_manager.h>
#include <libdistributed_work_queue_options.h>
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
#include <libpressio_ext/cpp/serializable.h>
#include <libpressio_ext/cpp/distributed_manager.h>

class many_independent_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "many_independent:compressor", compressor_id, compressor);
    set(options, "many_independent:input_data_groups",
        pressio_data(input_data_groups.begin(), input_data_groups.end()));
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
    pressio_data tmp;

    get_meta(options, "many_independent:compressor", compressor_plugins(), compressor_id, compressor);
    if(get(options, "many_independent:input_data_groups", &tmp) == pressio_options_key_set) {
      input_data_groups = tmp.to_vector<int>();
    }
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
    compressor->set_name(name + "/" + compressor->prefix());
    manager.set_name(name);
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
    using response_t = std::tuple<int, int, std::vector<pressio_data>>; //group_idx, status, data

    auto effective_input_group = normalize_data_group(input_data_groups, inputs.size());
    auto effective_output_group = normalize_data_group(output_data_groups, outputs.size());

    if(effective_input_group.size() != inputs.size()) {
      set_error(1, "invalid input group");
    }
    if(effective_output_group.size() != outputs.size()) {
      set_error(1, "invalid output group");
    }
    size_t num_groups = valid_data_groups(effective_input_group, effective_output_group);
    if(num_groups == 0) {
      set_error(2, "invalid data groups");
    }


    std::set<int> indicies(std::begin(effective_input_group), std::end(effective_input_group));

    int status = 0;
    manager.work_queue(
        std::begin(indicies), std::end(indicies),
        [&inputs, &outputs, &action, &effective_input_group, &effective_output_group](request_t request) {
          int idx = std::get<0>(request);
          auto input_data = make_data_group<pressio_data const*>(inputs, idx, effective_input_group);
          auto output_data_ptrs = make_data_group<pressio_data*>(outputs, idx, effective_output_group);

          int status = action(
              input_data.data(),
              input_data.data() + input_data.size(),
              output_data_ptrs.data(),
              output_data_ptrs.data() + output_data_ptrs.size()
              );
          std::vector<pressio_data> output_data;
          output_data.reserve(output_data_ptrs.size());
          for (auto output_data_ptr : output_data_ptrs) {
            output_data.emplace_back(std::move(*output_data_ptr));
          }


          return response_t{idx, status, std::move(output_data)};
        },
        [&outputs,&status,&effective_output_group](response_t response) {
          int idx = std::get<0>(response);
          status |= std::get<1>(response);
          auto output_data = std::move(std::get<2>(response));
          size_t out_idx=0;
          for (size_t i = 0; i < effective_output_group.size(); ++i) {
            if(effective_output_group[i] == idx) {
              *outputs[i] = std::move(output_data[out_idx++]);
            }
          }
        });
    return status;
  }

  template <class T, class Span>
  static std::vector<T> make_data_group(Span const& inputs, int idx, std::vector<int> const& data_groups) {
      std::vector<T> data_group;
      for (size_t i = 0; i < inputs.size(); ++i) {
        if(data_groups[i] == idx) {
          data_group.push_back(inputs[i]);
        }
      }
      return data_group;
  }

  /**
   *
   * \returns 0 on error, or the number of groups on success
   */
  size_t valid_data_groups(std::vector<int> const& effective_input_group, std::vector<int> const& effective_output_group) {
    std::set<int> s1(std::begin(effective_input_group), std::end(effective_input_group));
    std::set<int> s2(std::begin(effective_output_group), std::end(effective_output_group));
    return (s1 == s2)? s1.size(): 0;
  }

  std::vector<int> normalize_data_group(std::vector<int> const& data_group,  size_t size) {
    std::vector<int> ret;
    if(not data_group.empty()) {
      ret = data_group;
    } else {
      ret.resize(size);
      std::iota(std::begin(ret), std::end(ret), 0);
    }
    return ret;
  }


  std::vector<int> input_data_groups;
  std::vector<int> output_data_groups;

  pressio_distributed_manager manager = pressio_distributed_manager(
      /*max_masters*/1,
      /*max_ranks_per_worker*/pressio_distributed_manager::unlimited
      );
  pressio_compressor compressor = compressor_plugins().build("noop");
  std::string compressor_id = "noop";
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "many_independent", []() {
  return compat::make_unique<many_independent_compressor_plugin>();
});
