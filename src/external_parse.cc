#include "external_parse.h"
#include "pressio_version.h"
#include "libpressio_ext/cpp/options.h"
#include <sstream>

#if LIBPRESSIO_HAS_JSON
#include <nlohmann/json.hpp>
#include "libpressio_ext/cpp/json.h"
#endif

namespace libpressio { namespace launch {


    //returns the version number parsed, starts at 1, zero means error
    static std::string api_version_number(std::istringstream& stdout_stream)  {
      std::string version_line;
      std::getline(stdout_stream, version_line);
      auto eq_pos = version_line.find('=') + 1;
      if(version_line.substr(0, eq_pos) != "external:api=") {
        //report error
        throw std::runtime_error("invalid format version");
      }
      return version_line.substr(eq_pos);
    }

    void parse_v1(std::istringstream& stdout_stream, extern_proc_results const& input, pressio_options&proc_results_opts, bool is_default, std::string const& name, double duration, std::string const& result_prefix) {
      proc_results_opts.clear();

      //re-implment set here as non-member function
      auto set = [&](pressio_options& options, std::string const& key, pressio_option const& value) {
        if(name.empty()) options.set(key, value);
        else options.set(name, key, value);
      };

      std::map<std::string, std::vector<std::string>> values;
      for (std::string line; std::getline(stdout_stream, line); ) {
        auto equal_pos = line.find('=');
        std::string name = result_prefix + line.substr(0, equal_pos);
        std::string value_s = line.substr(equal_pos + 1);
        values[name].emplace_back(value_s);
      }
      for (auto const& i : values) {
        if (i.second.size() == 1) {
            auto const& value = i.second.front();
            try {
                set(proc_results_opts, i.first, std::stod(value));
            } catch (std::invalid_argument const&) {
                set(proc_results_opts, i.first, value);
            }
        } else {
            std::vector<double> values_as_double;
            try {
                std::transform(i.second.begin(), i.second.end(), std::back_inserter(values_as_double), [](std::string const& v) {
                        return std::stod(v);
                        });
                set(proc_results_opts, i.first, pressio_data(values_as_double.begin(), values_as_double.end()));
            } catch (std::invalid_argument const&) {
                set(proc_results_opts, i.first, i.second);
            }
        }
      }
      if(not is_default) {
        set(proc_results_opts, "external:stderr", input.proc_stderr);
        set(proc_results_opts, "external:return_code", input.return_code);
        set(proc_results_opts, "external:error_code", input.error_code);
        set(proc_results_opts, "external:runtime", duration);
      }
    }

#if LIBPRESSIO_HAS_JSON
    void parse_json(std::istringstream& stdout_stream, extern_proc_results const& input, pressio_options& results, bool is_default, std::string const& name, double duration, std::string const& result_prefix) {
      
      //re-implment set here as non-member function
      auto set = [&](pressio_options& options, std::string const& key, pressio_option const& value) {
        if(name.empty()) options.set(key, value);
        else options.set(name, key, value);
      };

      results.clear();

      nlohmann::json j;
      stdout_stream >> j;
      pressio_options options = j;

      for (auto const& item : options) {
        results.set(result_prefix+item.first, item.second);
      }

      if(not is_default) {
        set(results, "external:stderr", input.proc_stderr);
        set(results, "external:return_code", input.return_code);
        set(results, "external:error_code", input.error_code);
        set(results, "external:runtime", duration);
      }
    }
#endif

    size_t parse_result(extern_proc_results& proc_results, pressio_options& results, bool is_default, std::string const& name, double duration, std::string const& result_prefix) 
    {
      //re-implment set here as non-member function
      auto set = [&](pressio_options& options, std::string const& key, pressio_option const& value) {
        if(name.empty()) options.set(key, value);
        else options.set(name, key, value);
      };

      try{
        std::istringstream stdout_stream(proc_results.proc_stdout);
        auto api_version = api_version_number(stdout_stream);
        if(api_version == "1" || api_version == "2" || api_version == "3" || api_version == "4" || api_version == "5") {
            parse_v1(stdout_stream, proc_results, results, is_default, name, duration, result_prefix);
            return stoull(api_version);
#if LIBPRESSIO_HAS_JSON
        } else if(api_version == "json:1") {
            parse_json(stdout_stream, proc_results, results, is_default, name, duration, result_prefix);
            return 1;
#endif
        }

      } catch(...) {} //swallow all errors and set error information

      results.clear();
      if(not is_default) {
        set(results, "external:error_code", static_cast<int32_t>(format_error));
        set(results, "external:return_code", 0);
        set(results, "external:stderr", proc_results.proc_stderr);
        set(results, "external:runtime", duration);
      }
      return 0;
    }

} }
