#define SOL_ALL_SAFETIES_ON 1
#define SOL_PRINT_ERRORS 1
#include <chrono>
#include <stdexcept>
#include <sol/sol.hpp>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/libpressio.h"

namespace libpressio { namespace compressors { namespace lambda_fn_ns {

enum class lambda_fn_event {
  compress,
  decompress,
  set_options
};

class lambda_fn_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "lambda_fn:compressor",  compressor_id, compressor);
    set(options, "lambda_fn:script",  script);
    set(options, "lambda_fn:on_set_options",  on_set_options);
    set(options, "lambda_fn:on_compress",  on_compress);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "lambda_fn:compressor", compressor_plugins(), compressor);
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    
        std::vector<std::string> invalidations {"lambda_fn:script", "lambda_fn:on_set_options", "lambda_fn:on_compress"}; 
        std::vector<pressio_configurable const*> invalidation_children {&*compressor}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "lambda_fn:compressor", "compressor to configure dynamically", compressor);
    set(options, "pressio:description", R"(a meta compressor that reconfigures the child compressor dynamically at runtime)");
    set(options, "lambda_fn:script", R"(The actual script to use just before the child compressor in invoked written in lua.

    When the script begins several global variables are defined

    + persist -- state persisted between calls to script
    + inputs -- the list of inputs provided to the compress or decompress function
    + outputs -- the list of outputs provided to the compress or decompress function
    + set_options -- the list of options provided to the set_options function
    + options -- the list of options to apply to the child compressor
    + name -- a string containing the name of this lambda_fn
    + is_compress -- true if compression is the next operation
    + is_decompress -- true if decompression is the next operation
    + is_set_options -- true if set_options is the next operation

    The lua base and math libraries are exposed by default, this may change in the future to allow
    the user to specify which libraries are allowed

    A number of libpressio enums with all of their values are exposed to lua

    pressio_dtype
    pressio_option_type
    pressio_options_key_status

    Additionally a number of libpressio types are exposed to lua with a subset of their api

    `pressio_data`

    + `new()` -- constructs an empty pressio_data object
    + `new(dtype, dims, data)` -- constructs a pressio_data object with lua tables dims and data
    + `dimensions()` -- return a vector of the dimensions
    + `dtype()` -- return the dtype type of the data object
    + `to_vector()` -- convert the data object to a 1d vector that can be manipulated

    `pressio_option`

    + `new()` -- construct an empty option
    + `type()` -- get the current type
    + `get_int8()` -- get the value as an int8
    + `get_int16()` -- get the value as an int16
    + `get_int32()` -- get the value as an int32
    + `get_int64()` -- get the value as an int64
    + `get_uint8()` -- get the value as an uint8
    + `get_uint16()` -- get the value as an uint16
    + `get_uint32()` -- get the value as an uint32
    + `get_uint64()` -- get the value as an uint64
    + `get_float()` -- get the value as an float
    + `get_double()` -- get the value as an double
    + `get_string()` -- get the value as a string
    + `get_string_arr()` -- get the value as a string array
    + `get_data()` -- get the value as a pressio_data object
    + `set_int8()` -- set the value as an int8
    + `set_int16()` -- set the value as an int16
    + `set_int32()` -- set the value as an int32
    + `set_int64()` -- set the value as an int64
    + `set_uint8()` -- set the value as an uint8
    + `set_uint16()` -- set the value as an uint16
    + `set_uint32()` -- set the value as an uint32
    + `set_uint64()` -- set the value as an uint64
    + `set_float()` -- set the value as an float
    + `set_double()` -- set the value as an double
    + `set_string()` -- set the value as a string
    + `set_string_arr()` -- set the value as a string array
    + `set_data()` -- set the value as a pressio_data object

    `pressio_options`

    + `key_status()` -- get the status of a key
    + `get()` -- get the pressio_option for a value
    + `set()` -- set the pressio_option for a value

    example 1:

    ```lua
    -- use compressor 0 for floats and compressor 1 for integers
    local is_floating = false;
    local dtype = pressio_dtype.int8;
    if is_compress then
      dtype = inputs[1]:dtype();
    else
      dtype = outputs[1]:dtype();
    end

    if dtype == pressio_dtype.float or dtype == pressio_dtype.double then
      local option = pressio_option:new()
      option:set_uint64(0)
      options:set("switch:active_id", option)
    else
      local option = pressio_option:new()
      option:set_uint64(1)
      options:set("switch:active_id", option)
    end
    ```

    example 2:

    ```lua
    -- chunk the data by last two dimensions, assumes at least 2d data
    if is_compress then
      local dims = inputs[1]:dimensions();
      local dims_size = #dims;
      local chunk_sizes = {};

      for i=1,dims_size do
        chunk_sizes[i] = 1;
      end
      chunk_sizes[dims_size-1] = dims[dims_size-1]
      chunk_sizes[dims_size] = dims[dims_size]

      local chunk_data = pressio_data.new(pressio_dtype.uint64, {dims_size}, chunk_sizes)
      local chunk_size_op = pressio_option:new();
      chunk_size_op:set_data(chunk_data);

      options:set("chunking:size", chunk_size_op)
    end
    ```
    )");
    set(options, "lambda_fn:on_set_options", R"(nonzero if the script should be called at the end of set_options,
        takes effect as soon as it is set)");
    set(options, "lambda_fn:on_compress", R"(nonzero if the script should be called just before the child compressor is invoked,
        takes effect as soon as it is set)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "lambda_fn:compressor", compressor_plugins(), compressor_id, compressor);
    get(options, "lambda_fn:script", &script);
    get(options, "lambda_fn:on_set_options",  &on_set_options);
    get(options, "lambda_fn:on_compress",  &on_compress);
    if(on_set_options) {
      run_options_script(options, lambda_fn_event::set_options);
    }
    return 0;
  }

  int compress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs) override 
  {
    try { 
      run_compress_script(inputs, outputs, lambda_fn_event::compress);
      int ret = compressor->compress_many(inputs.data(), inputs.size(), outputs.data(), outputs.size());
      if(ret) {
        set_error(ret, compressor->error_msg());
      }
      return ret;
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

  int decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs) override 
  {
    try {
      run_compress_script(inputs, outputs, lambda_fn_event::decompress);
      int ret = compressor->decompress_many(inputs.data(), inputs.size(), outputs.data(), outputs.size());
      if(ret) {
        set_error(ret, compressor->error_msg());
      }
      return ret;
    } catch (std::exception const& ex) {
      return set_error(2, ex.what());
    }
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

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "lambda_fn"; }

  pressio_options get_metrics_results_impl() const override {
    pressio_options opts;
    opts.copy_from(compressor->get_metrics_results());
    set(opts, "lambda_fn:compress_script_time", compress_script_time);
    set(opts, "lambda_fn:decompress_script_time", decompress_script_time);
    return opts;
  }

  void set_name_impl(std::string const& name) override {
    if(name != "") {
      compressor->set_name(name + "/" + compressor->prefix());
    } else {
      compressor->set_name(name );
    }
  }
  std::vector<std::string> children_impl() const final {
      return { compressor->get_name() };
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<lambda_fn_compressor_plugin>(*this);
  }

private:

  void bind_pressio(sol::state& lua) 
  {
    lua.new_enum<pressio_dtype>("pressio_dtype", {
          {"int8", pressio_int8_dtype},
          {"int16", pressio_int16_dtype},
          {"int32", pressio_int32_dtype},
          {"int64", pressio_int64_dtype},
          {"uint8", pressio_uint8_dtype},
          {"uint16", pressio_uint16_dtype},
          {"uint32", pressio_uint32_dtype},
          {"uint64", pressio_uint64_dtype},
          {"float", pressio_float_dtype},
          {"double", pressio_double_dtype},
        });
    lua.new_enum<pressio_option_type>("pressio_option_type", {
          {"uint32", pressio_option_uint32_type},
          {"int32", pressio_option_int32_type},
          {"float", pressio_option_float_type},
          {"double", pressio_option_double_type},
          {"charptr", pressio_option_charptr_type},
          {"usrptr", pressio_option_userptr_type},
          {"unset", pressio_option_unset_type},
          {"charptr_arr", pressio_option_charptr_array_type},
          {"data", pressio_option_data_type},
          {"int8", pressio_option_int8_type},
          {"uint8", pressio_option_uint8_type},
          {"int16", pressio_option_int16_type},
          {"uint16", pressio_option_uint16_type},
          {"int64", pressio_option_int64_type},
          {"uint64", pressio_option_uint64_type},
        });
    lua.new_enum<pressio_options_key_status>("pressio_options_key_status", {
          {"set", pressio_options_key_set},
          {"does_not_exist", pressio_options_key_does_not_exist},
          {"exists", pressio_options_key_exists},
        });
    lua.new_usertype<pressio_data>(
        "pressio_data",
        sol::meta_function::construct,
        sol::factories(
          [](int dtype, sol::table const& dims, sol::table const& data){
            std::vector<size_t> real_dims;
            std::vector<double> real_data;
            for (auto const& i : dims) {
              real_dims.emplace_back(i.second.as<size_t>());
            }
            for (auto const& i : data) {
              real_data.emplace_back(i.second.as<double>());
            }

            auto lp_data = pressio_data(real_data.begin(), real_data.end()).cast(static_cast<pressio_dtype>(dtype));
            lp_data.reshape(real_dims);
            return lp_data;
          },
          []{pressio_data data; return data;}
          ),
        "dimensions", &pressio_data::dimensions,
        "dtype", &pressio_data::dtype,
        "to_vector", &pressio_data::to_vector<double>
        );
    lua.new_usertype<pressio_options>(
        "pressio_options",
        "key_status", static_cast<pressio_options_key_status (pressio_options::*)(std::string const&) const>(&pressio_options::key_status),
        "get", static_cast<pressio_option const& (pressio_options:: *)(std::string const&) const>(&pressio_options::get),
        "set", static_cast<void (pressio_options::*)(std::string const&, pressio_option const&)>(&pressio_options::set)
        );
    lua.new_usertype<pressio_option>(
        "pressio_option",
        "type", &pressio_option::type,
        "get_int8", &pressio_option::get_value<int8_t>,
        "get_int16", &pressio_option::get_value<int16_t>,
        "get_int32", &pressio_option::get_value<int32_t>,
        "get_int64", &pressio_option::get_value<int64_t>,
        "get_uint8", &pressio_option::get_value<uint8_t>,
        "get_uint16", &pressio_option::get_value<uint16_t>,
        "get_uint32", &pressio_option::get_value<uint32_t>,
        "get_uint64", &pressio_option::get_value<uint64_t>,
        "get_float", &pressio_option::get_value<float>,
        "get_double", &pressio_option::get_value<double>,
        "get_string", &pressio_option::get_value<std::string>,
        "get_string_arr", &pressio_option::get_value<std::vector<std::string>>,
        "get_data", &pressio_option::get_value<pressio_data>,
        "set_int8", &pressio_option::set<int8_t>,
        "set_int16", &pressio_option::set<int16_t>,
        "set_int32", &pressio_option::set<int32_t>,
        "set_int64", &pressio_option::set<int64_t>,
        "set_uint8", &pressio_option::set<uint8_t>,
        "set_uint16", &pressio_option::set<uint16_t>,
        "set_uint32", &pressio_option::set<uint32_t>,
        "set_uint64", &pressio_option::set<uint64_t>,
        "set_float", &pressio_option::set<float>,
        "set_double", &pressio_option::set<double>,
        "set_string", &pressio_option::set<std::string>,
        "set_string_arr", &pressio_option::set<std::vector<std::string> const&>,
        "set_data", &pressio_option::set<pressio_data const&>
        );


  }

  void run_options_script(pressio_options const& set_opts, lambda_fn_event event) {
    run_script_common(event, [&set_opts](sol::state& lua){
        lua["set_options"] = std::ref(set_opts);
    });
  }
  void run_compress_script(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs, lambda_fn_event event) {
    //TODO provide a way to run the script that moves the data to the host
    auto time = run_script_common(event, [&inputs, &outputs](sol::state& lua){
        lua["inputs"] = std::ref(inputs);
        lua["outputs"] = std::ref(outputs);
    });
    switch(event) {
      case lambda_fn_event::compress:
        compress_script_time = time;
        break;
      case lambda_fn_event::decompress:
        decompress_script_time = time;
        break;
      default:
        break;
    }
  }

  template <class Func>
  uint64_t run_script_common(lambda_fn_event event, Func&& func) {
    pressio_options opts;
    auto begin = std::chrono::steady_clock::now();

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    bind_pressio(lua);
    lua["persist"] = std::ref(persist);
    lua["options"] = std::ref(opts);
    lua["name"] = get_name();
    lua["is_compress"] = event == lambda_fn_event::compress;
    lua["is_decompress"] = event == lambda_fn_event::decompress;
    lua["is_set_options"] = event == lambda_fn_event::set_options;
    func(lua);

    lua.safe_script(script);

    auto ret = compressor->set_options(opts); 

    auto end = std::chrono::steady_clock::now();
    if(ret > 0) {
      throw std::runtime_error(error_msg());
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
  }

  std::string script;
  pressio_compressor compressor = compressor_plugins().build("noop");
  std::string compressor_id = "noop";
  uint64_t compress_script_time = 0;
  uint64_t decompress_script_time = 0;
  pressio_options persist;
  int32_t on_compress=1, on_set_options=0;
};

pressio_register registration(compressor_plugins(), "lambda_fn", []() {
  return compat::make_unique<lambda_fn_compressor_plugin>();
});

} }}
