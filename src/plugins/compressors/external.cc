#include <tuple>
#include <regex>
#include <chrono>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/printers.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/launch/external_launch.h"
#include "external_parse.h"
#include "cleanup.h"
#include "pressio_posix.h"
#include <unistd.h>

#include <nlohmann/json.hpp>
#include "libpressio_ext/cpp/json.h"

namespace libpressio { namespace external_compressor_ns {

class external_compressor_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct io_config {
      std::vector<std::string> field_names{""}, prefixes{""}, suffixes{""};
      std::vector<std::string> io_str {"posix"};
      std::vector<pressio_io> io {io_plugins().build("posix")};
      bool write = false;
      bool use_template = true;
  };

  pressio_options invoke_settings(std::string const& mode) const {

      std::vector<std::string> args{
          "--api",
          "5",
          "--mode",
          mode,
          "--config_name",
          config_name,
      };
      if(!get_name().empty()) {
          args.emplace_back("--name");
          args.emplace_back(get_name());
      }
      if(external_options.size() != 0) {
          args.push_back("--options");
          nlohmann::json jstr = external_options;
          std::string str = jstr.dump();
          args.push_back(str);
      }
      auto start_time = std::chrono::high_resolution_clock::now();
      auto result = launch->launch(args);
      auto end_time = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end_time - start_time).count();
      pressio_options ret;

      parse_result(result, ret, false, get_name(), duration, "");
      return ret;
  }
  int invoke_compress(std::string const& mode,compat::span<const pressio_data* const> const& input_datas, compat::span<struct pressio_data*>& outputs, io_config& in_config, io_config& out_config) {

      std::vector<std::string> args{
          "--api",
          "5",
          "--mode",
          mode,
          "--config_name",
          config_name,
      };
      if(!name.empty()) {
          args.emplace_back("--name");
          args.emplace_back(get_name());
      }
      auto format_flag = [](std::string const& flag, std::string const& field_name) {
          if(field_name.empty()) {
              std::stringstream ss;
              ss << "--" << flag;
              return ss.str();
          } else {
              std::stringstream ss;
              ss << "--" << field_name << "_" << flag;
              return ss.str();
          }
      };
      if(external_options.size() != 0) {
          args.push_back("--options");
          nlohmann::json jstr = external_options;
          std::string str = jstr.dump();
          args.push_back(str);
      }
      std::vector<cleanup> cleanup_files;
      for (size_t i = 0; i < input_datas.size(); ++i) {
          auto const& data = input_datas[i];

          args.push_back(format_flag("itype",in_config.field_names[i]));
          std::stringstream type_ss;
          type_ss << data->dtype();
          args.push_back(type_ss.str());
          
          for (auto d : data->dimensions()) {
              args.push_back(format_flag("idim", in_config.field_names[i]));
              args.push_back(std::to_string(d));
          }

          std::ostringstream ss;
          if(in_config.prefixes.size() > i) {
              ss << in_config.prefixes[i];
          }
          ss << ".pressioinXXXXXX";
          int suffix_len = 0;
          if(in_config.suffixes.size() > i) {
              ss << in_config.suffixes[i];
              suffix_len = static_cast<int>(in_config.suffixes[i].size());
          }
          std::string input_fd_name(ss.str());
          int input_fd = mkstemps(&input_fd_name[0], suffix_len);
          if(input_fd == -1) {
              set_error(errno, errno_to_error());
          }
          char* resolved_input = realpath(input_fd_name.c_str(), nullptr);
          input_fd_name = resolved_input;
          free(resolved_input);
          cleanup_files.emplace_back([input_fd, input_fd_name]{
              close(input_fd);
              unlink(input_fd_name.c_str());
          });

          in_config.io[i]->set_options({{"io:path", input_fd_name}});
          args.push_back(format_flag("input", in_config.field_names[i]));
          args.push_back(input_fd_name);
          if(in_config.io[i]->write(data)) {
              return set_error(in_config.io[i]->error_code(), in_config.io[i]->error_msg());
          }
      }

      for (size_t i = 0; i < outputs.size(); ++i) {
          auto const& data = outputs[i];
          
          args.push_back(format_flag("otype", out_config.field_names[i]));
          std::stringstream type_ss;
          type_ss << data->dtype();
          args.push_back(type_ss.str());
          
          for (auto d : data->dimensions()) {
              args.push_back(format_flag("odim", out_config.field_names[i]));
              args.push_back(std::to_string(d));
          }

          if(data->has_data() && compressed.write) {
              //do write include path
              std::ostringstream ss;
              if(in_config.prefixes.size() > i) {
                  ss << out_config.prefixes[i];
              }
              ss << ".pressioinXXXXXX";
              int suffix_len = 0;
              if(out_config.suffixes.size() > i) {
                  ss << out_config.suffixes[i];
                  suffix_len = static_cast<int>(out_config.suffixes[i].size());
              }
              std::string input_fd_name(ss.str());
              int input_fd = mkstemps(&input_fd_name[0], suffix_len);
              if(input_fd == -1) {
                  set_error(errno, errno_to_error());
              }
              char* resolved_input = realpath(input_fd_name.c_str(), nullptr);
              input_fd_name = resolved_input;
              free(resolved_input);
              cleanup_files.emplace_back([input_fd, input_fd_name]{
                  close(input_fd);
                  unlink(input_fd_name.c_str());
              });

              args.push_back(format_flag("output", out_config.field_names[i]));
              args.push_back(input_fd_name);
          }
      }


      auto start_time = std::chrono::high_resolution_clock::now();
      auto result = launch->launch(args);
      auto end_time = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end_time - start_time).count();
      pressio_options ret;

      parse_result(result, ret, false, get_name(), duration, "");
      int ec, rc;
      get(ret, "external:return_code", &rc);
      get(ret, "external:error_code", &ec);

      for (size_t i = 0; i < outputs.size(); ++i) {
          std::ostringstream sspath;
          sspath << "output:" << i << ":path";
          std::string path_key = sspath.str();
          std::string path;
          if(get(ret, path_key, &path) != pressio_options_key_set) {
              return set_error(1, "expected key " + path_key + " not found");
          }

          std::ostringstream ssdims;
          ssdims << "output:" << i << ":dims";
          std::string dims_key = ssdims.str();
          pressio_data dims_data;
          if(get(ret, dims_key, &dims_data) != pressio_options_key_set) {
              return set_error(1, "expected key " + dims_key + " not found");
          }
          auto dims = dims_data.to_vector<size_t>();


          std::ostringstream ssdtype;
          ssdtype << "output:" << i << ":dtype";
          std::string dtype_key = ssdtype.str();
          pressio_dtype dtype;
          if(ret.cast(dtype_key, &dtype, pressio_conversion_special) != pressio_options_key_set)  {
              return set_error(1, "expected key " + dtype_key + " not found");
          }

          out_config.io[i]->set_options({{"io:path", path}});
          if(out_config.use_template) {
              auto data_template = pressio_data::owning(dtype, dims);
              pressio_data* ptr = out_config.io[i]->read(&data_template);
              (*outputs[i]) = std::move(*ptr);
              delete ptr;
          } else {
              pressio_data* ptr = out_config.io[i]->read(nullptr);
              (*outputs[i]) = std::move(*ptr);
              delete ptr;
          }
      }

      std::string key;
      pressio_option option;
      const std::regex metrics_pattern{"metric:(.+)"};
      std::smatch match;
      for (auto const& i : ret) {
          std::tie(key,option) = i;
          if(std::regex_match(key, match, metrics_pattern)) {
              auto const& entry = match[1];
              set(external_metrics, entry, option);
          }
      }

      if(ec?ec:rc) {
          std::string error;
          get(ret, "external:stderr", &error);
          return set_error(ec?ec:rc, error);
      }

      return 0;
  }

  struct pressio_options get_options_impl() const override
  {
    pressio_options options;
    set_meta(options, "external_compressor:launch", launch_str, launch);
    auto set_ioconfig = [&options,this](io_config const& config, std::string const& key) {
        set_meta_many(options, "external_compressor:" + key + "_io" , config.io_str, config.io);
        set(options, "external_compressor:" + key + "_feild_names" , config.field_names);
        set(options, "external_compressor:" + key + "_prefixes" , config.prefixes);
        set(options, "external_compressor:" + key + "_suffixes" , config.suffixes);
        set(options, "external_compressor:" + key + "_do_write" , config.write);
    };
    set_ioconfig(full, "uncompressed");
    set_ioconfig(compressed, "compressed");
    options.copy_from(invoke_settings("get_options"));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "external_compressor:launch", launch_plugins(), launch);
    set_meta_many_configuration(options, "external_compressor:uncompressed_io", io_plugins(), full.io);
    set_meta_many_configuration(options, "external_compressor:compressed_io", io_plugins(), compressed.io);
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    options.copy_from(invoke_settings("get_configuration"));
    
        std::vector<std::string> invalidations {}; 
        std::vector<pressio_configurable const*> invalidation_children {&*launch}; 
        
        for (auto const& child : full.io) {
            invalidation_children.emplace_back(&*child);
        }
        for (auto const& child : compressed.io) {
            invalidation_children.emplace_back(&*child);
        }
                
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    auto set_ioconfig = [&options, this](io_config const& config, std::string const& key) {
        set_meta_many_docs(options, "external_compressor:" + key + "_io" , "io to write " + key + " values", config.io);
        set(options, "external_compressor:" + key + "_feild_names" , "field names for " + key + " values");
        set(options, "external_compressor:" + key + "_prefixes" , "prefixes for " + key + " values");
        set(options, "external_compressor:" + key + "_suffixes" , "suffixes for " + key + " values");
        set(options, "external_compressor:" + key + "_do_write" , "enable writing buffers for " + key + " values");
    };
    set_ioconfig(full, "uncompressed");
    set_ioconfig(compressed, "compressed");
    options.copy_from(invoke_settings("get_documentation"));
    set_meta_docs(options, "external_compressor:launch", "method to launch external compressor", launch);
    set(options, "pressio:description", R"(invoke an external compressor)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    external_options = options;
    get_meta(options, "external_compressor:launch", launch_plugins(), launch_str, launch);
    auto set_ioconfig = [&options,this](io_config& config, std::string const& key) {
        get_meta_many(options, "external_compressor:" + key + "_io" , io_plugins(), config.io_str, config.io);
        get(options, "external_compressor:" + key + "_feild_names" , &config.field_names);
        get(options, "external_compressor:" + key + "_prefixes" , &config.prefixes);
        get(options, "external_compressor:" + key + "_suffixes" , &config.suffixes);
        get(options, "external_compressor:" + key + "_do_write" , &config.write);
    };
    set_ioconfig(full, "uncompressed");
    set_ioconfig(compressed, "compressed");

    auto result = invoke_settings("set_options");
    int rc, ec;
    get(result, "external:return_code", &rc);
    get(result, "external:error_code", &ec);

    if(ec?ec:rc) {
        std::string error;
        get(result, "external:stderr", &error);
        return set_error(ec?ec:rc, error);
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
      compat::span<const pressio_data* const> const input_datas(&input, &input+1);
      compat::span<struct pressio_data*> outputs(&output, &output+1);
      return compress_many_impl(input_datas, outputs);
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
      compat::span<const pressio_data* const> const input_datas(&input, &input+1);
      compat::span<struct pressio_data*> outputs(&output, &output+1);
      return decompress_many_impl(input_datas, outputs);
  }

  int compress_many_impl(compat::span<const pressio_data* const> const& input_datas,
                    compat::span<struct pressio_data*>& outputs) override
  {
      return invoke_compress("compress", input_datas, outputs, full, compressed);
  }

  int decompress_many_impl(const compat::span<pressio_data const*const>& inputs, compat::span<struct pressio_data*>& outputs) override {
      return invoke_compress("decompress", inputs, outputs, compressed, full);
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "external_compressor"; }

  pressio_options get_metrics_results_impl() const override {
    return external_metrics;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<external_compressor_compressor_plugin>(*this);
  }

  pressio_options external_options, external_metrics;
  io_config full, compressed;
  std::string config_name = "external";
  std::string launch_str = "forkexec";
  pressio_launcher launch = launch_plugins().build(launch_str);
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "external_compressor", []() {
  return compat::make_unique<external_compressor_compressor_plugin>();
});

} }
