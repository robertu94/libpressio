#ifndef LIBPRESSSIO_SUBGROUP_MANAGER
#define LIBPRESSSIO_SUBGROUP_MANAGER
#include "configurable.h"
#include "std_compat/memory.h"
#include "std_compat/span.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_compressor.h"
#include "pressio_data.h"
#include "pressio_options.h"
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <set>

/** 
 *  \file
 *  \brief helper for subgroups in meta-objects which support multiple inputs 
 * */

/**
 * a helper class to map input and output buffers to a meta-compressor
 */
class pressio_subgroup_manager: public pressio_configurable, public pressio_errorable {
public:
  int set_options(const struct pressio_options &options) override {
    pressio_data tmp;
    if(get(options, "subgroups:input_data_groups", &tmp) == pressio_options_key_set) {
      input_data_groups = tmp.to_vector<int>();
    }
    if(get(options, "subgroups:output_data_groups", &tmp) == pressio_options_key_set) {
      output_data_groups = tmp.to_vector<int>();
    }
    return 0;
  }
  struct pressio_options get_documentation() const override {
    pressio_options opts;
    set(opts, "subgroups:input_data_groups", "which inputs get mapped to which sub compression operations");
    set(opts, "subgroups:output_data_groups", "which outputs get mapped to which sub compression operations");
    return opts;
  }
  struct pressio_options get_options() const override {
    pressio_options opts;
    set(opts, "subgroups:input_data_groups", pressio_data(std::begin(input_data_groups), std::end(input_data_groups)));
    set(opts, "subgroups:output_data_groups", pressio_data(std::begin(output_data_groups), std::end(output_data_groups)));
    return opts;
  }

  const char* prefix() const override {
    return "subgroups";
  }

  /**
   * makes the input and output groups match sizes and other sanity tests
   *
   * \param[in] inputs the inputs groups
   * \param[in] outputs the output groups
   */
  template <class U, class V>
  int normalize_and_validate(compat::span<U> const& inputs, compat::span<V> const& outputs) {
    effective_input_group = normalize_data_group(input_data_groups, inputs.size());
    effective_output_group = normalize_data_group(output_data_groups, outputs.size());

    if(effective_input_group.size() != inputs.size()) {
      return set_error(1, "invalid input group");
    }
    if(effective_output_group.size() != outputs.size()) {
      return set_error(1, "invalid output group");
    }
    size_t num_groups = valid_data_groups(effective_input_group, effective_output_group);
    if(num_groups == 0) {
      return set_error(2, "invalid data groups");
    }
    return 0;
  }

  /**
   * \param[in] inputs the actual inputs
   * \param[in] idx which input group to retrieve
   * \returns the input group based on internal configuration
   */
  template <class Span>
  std::vector<pressio_data const*> get_input_group(Span const& inputs, int idx) const {
    return make_data_group<pressio_data const*>(inputs, idx, effective_input_group);
  }

  /**
   * \param[in] inputs the actual inputs
   * \param[in] idx which input group to retrieve
   * \returns the input group based on internal configuration
   */
  template <class Span>
  std::vector<pressio_data*> get_output_group(Span const& inputs, int idx) const {
    return make_data_group<pressio_data*>(inputs, idx, effective_output_group);
  }

  /**
   * \returns the effective input groups
   */
  std::vector<int> const& effective_input_groups() const {
    return effective_input_group;
  }

  /**
   * \returns the effective outputs groups
   */
  std::vector<int> const& effective_output_groups() const {
    return effective_output_group;
  }


private:

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
  static std::vector<int> normalize_data_group(std::vector<int> const& data_group,  size_t size) {
    std::vector<int> ret;
    if(not data_group.empty()) {
      ret = data_group;
    } else {
      ret.resize(size);
      std::iota(std::begin(ret), std::end(ret), 0);
    }
    return ret;
  }
  /**
   *
   * \returns 0 on error, or the number of groups on success
   */
  static size_t valid_data_groups(std::vector<int> const& effective_input_group, std::vector<int> const& effective_output_group) {
    std::set<int> s1(std::begin(effective_input_group), std::end(effective_input_group));
    std::set<int> s2(std::begin(effective_output_group), std::end(effective_output_group));
    return (s1 == s2)? s1.size(): 0;
  }

  std::vector<int> effective_input_group, effective_output_group;
  std::vector<int> input_data_groups, output_data_groups;
};

#endif /* end of include guard: LIBPRESSSIO_SUBGROUP_MANAGER */
