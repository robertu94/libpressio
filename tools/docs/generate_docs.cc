#include <libpressio.h>
#include <libpressio_ext/io/pressio_io.h>
#include <libpressio_ext/cpp/printers.h>
#include <set>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <regex>
#include <std_compat/string_view.h>
#include <unistd.h>

const char* name = "pressio";

template <class Func>
std::vector<std::string> metas_list(Func&& get_supported_meta) {
  auto* instance = pressio_instance();
  const std::string meta_str = get_supported_meta();
  std::vector<std::string> metas;
  size_t begin = 0;
  size_t last_space = meta_str.find_first_of(' ');
  while(last_space != std::string::npos) {
    metas.emplace_back(meta_str.substr(begin, last_space -begin));
    begin = last_space + 1;
    last_space = meta_str.find_first_of(' ', begin+1);
  }
  pressio_release(instance);

  std::sort(
      std::begin(metas),
      std::end(metas),
      [](std::string const& lhs, std::string const& rhs) {
        std::string lowered_lhs(lhs);
        std::transform(lowered_lhs.begin(), lowered_lhs.end(), lowered_lhs.begin(), ::tolower);
        std::string lowered_rhs(rhs);
        std::transform(lowered_rhs.begin(), lowered_rhs.end(), lowered_rhs.begin(), ::tolower);
        return lowered_lhs < lowered_rhs;
      });

  return metas;
}

std::string
trim(std::string const & source) {
  std::regex r{"\n[ \t]*"};
  std::ostringstream os;
  std::regex_replace(std::ostreambuf_iterator<char>(os),
      source.begin(), source.end(), r, "\n");
  return os.str();
}

template <class Func>
void for_each_options(struct pressio_options* options, Func f) {
    pressio_options_iter* iter = pressio_options_get_iter(options);
    while(pressio_options_iter_has_value(iter)) {
      const char* key = pressio_options_iter_get_key(iter);
      pressio_option* value = pressio_options_iter_get_value(iter);
      f(key, value);
      pressio_option_free(value);
      pressio_options_iter_next(iter);
    }
    pressio_options_iter_free(iter);
}

void document_compressor(std::ostream& out, std::string const& compressor_id, const pressio_compressor* c) {
    pressio_options* docs = pressio_compressor_get_documentation(c);
    pressio_options* options = pressio_compressor_get_options(c);
    pressio_options* configuration = pressio_compressor_get_configuration(c);

//    out << *configuration << std::endl;
//    out << *options << std::endl;
//    out << *docs << std::endl;

    const char* stability = nullptr;
    const char* description=nullptr;
    int32_t thread_safety = 0;
    pressio_options_get_string(configuration, "/pressio:pressio:stability", &stability);
    pressio_options_get_string(docs, "/pressio:pressio:description", &description);
    pressio_options_get_integer(configuration, "/pressio:pressio:thread_safe", &thread_safety);

    //print the id first and version information
    out << "## " << compressor_id << std::endl;
    out << std::endl;
    out << "version: " << pressio_compressor_version(c) << std::endl << std::endl;
    out << "stability: " << std::string(stability, (stability == nullptr)? 0: strlen(stability)) << std::endl << std::endl;
    out << "thread_safety: " << pressio_thread_safety(thread_safety) << std::endl;
    out << std::endl;


    //then print the pressio:description entry
    out << trim(std::string(description, (description == nullptr)? 0: strlen(description))) << std::endl << std::endl;
    free((char*)stability);
    free((char*)description);
    
    //then print the remaining entries in the docs in a table
    bool first_option = true;
    std::set<std::string> skip_list;

    for_each_options(options, [&](const char* key, pressio_option* value) {
      if(std::string(key).find("/pressio:") == 0) {
        const std::string key_name = std::string(key).substr(strlen(name) + 2);

        if(key_name == "metrics:copy_compressor_results") {
          skip_list.emplace(key);
          return;
        }
        if(key_name == "metrics:errors_fatal") {
          skip_list.emplace(key);
          return;
        }

        if(first_option) {
          out << "### Options" << std::endl;
          first_option = false;
        }


        //get the description from the docs
        const char* func_description = nullptr;
        pressio_options_get_string(docs, key, &func_description);

        pressio_option_type type = pressio_option_get_type(value);

        out << "#### " << key_name << std::endl;
        out <<  std::endl;
        out << "type: `" << type << '`' << std::endl << std::endl;
        if(value->has_value()) {
          out << "default: " << '`' << *value << '`' << std::endl << std::endl;
        } else {
          out << "unset by default" << std::endl << std::endl;
        }
        //check configuration for enumerations
        const char** entries;
        size_t n_entries = 0;
        if(pressio_options_get_strings(configuration, key, &n_entries, &entries) == pressio_options_key_set) {
          skip_list.emplace(key);
          out << "options: ";
          for (int i = 0; i < n_entries; ++i) {
            out << entries[i];
            free((char*)entries[i]);
            if(i != n_entries -1) {
              out << ", ";
            }
          }
          free((char*)entries);
          out << std::endl << std::endl;
        }


        out << "description: " << trim(std::string(func_description, (func_description == nullptr)? 0: strlen(func_description)));
        out << std::endl;
        out << std::endl;

        free((char*)func_description);
      }
    });


    out << std::endl;
    bool first_config = true;

    for_each_options(configuration, [&](const char* key, pressio_option* value) {
      if(std::string(key).find("/pressio:") == 0) {
        const std::string key_name = std::string(key).substr(strlen(name) + 2);
        if(key_name == "pressio:thread_safe") {
          skip_list.emplace(key);
          return;
        }
        if(key_name == "pressio:stability") {
          skip_list.emplace(key);
          return;
        }

        if(skip_list.find(key) != skip_list.end()) {
          return;
        }

        if(first_config) {
          out << "### Configuration" << std::endl;
          first_config = false;
        }

        const char* func_description = nullptr;
        pressio_options_get_string(docs, key, &func_description);

        pressio_option_type type = pressio_option_get_type(value);

        out << "#### " << std::string(key).substr(strlen(name) + 2) << std::endl;
        out <<  std::endl;
        out << "type: `" << type << '`' << std::endl << std::endl;
        out << "value: " << '`' << *value << '`' << std::endl << std::endl;

        out << "description: " << trim(std::string(func_description, (func_description == nullptr)? 0: strlen(func_description)));
        out << std::endl;
        out << std::endl;

        free((char*)func_description);

      }
    });

    if(skip_list.size()) {
      if(first_config) {
        out << "### Configuration" << std::endl;
        first_config = false;
      }

      out << "#### other configuration entries: " << std::endl;
      for (auto const& i : skip_list) {
        out << "+ " << i.substr(strlen(name) + 2) << std::endl;
      }
    }
    out << std::endl;
    out << std::endl;

    pressio_options_free(docs);
    pressio_options_free(options);
    pressio_options_free(configuration);
}

void document_metrics(std::ostream& out, const char* metric, pressio_metrics const* c) {
    pressio_options* docs = pressio_metrics_get_documentation(c);
    pressio_options* options = pressio_metrics_get_options(c);
    pressio_options* configuration = pressio_metrics_get_configuration(c);
    pressio_options* metrics_results = pressio_metrics_get_results(c);

//    out << *configuration << std::endl;
//    out << *options << std::endl;
//    out << *docs << std::endl;

    const char* stability = nullptr;
    const char* description=nullptr;
    int32_t thread_safety = 0;
    pressio_options_get_string(configuration, "/pressio:pressio:stability", &stability);
    pressio_options_get_string(docs, "/pressio:pressio:description", &description);
    pressio_options_get_integer(configuration, "/pressio:pressio:thread_safe", &thread_safety);

    //print the id first and version information
    out << "## " << metric << std::endl;
    out << std::endl;
    out << "stability: " << std::string(stability, (stability == nullptr)? 0: strlen(stability)) << std::endl << std::endl;
    out << "thread_safety: " << pressio_thread_safety(thread_safety) << std::endl;
    out << std::endl;

    //then print the pressio:description entry
    out << trim(std::string(description, (description == nullptr)? 0: strlen(description))) << std::endl << std::endl;

    free((char*)stability);
    free((char*)description);

    bool first_metric_result = true;
    for_each_options(metrics_results, [&](const char* key, pressio_option* value) {
      if(std::string(key).find("/pressio:") == 0) {
        const std::string key_name = std::string(key).substr(strlen(name) + 2);

        if(first_metric_result) {
          out << "### Metrics Results" << std::endl;
          first_metric_result = false;
        }

        const char* func_description = nullptr;
        pressio_options_get_string(docs, key, &func_description);
        pressio_option_type type = pressio_option_get_type(value);

        out << "#### " << key_name << std::endl;
        out <<  std::endl;
        out << "type: `" << type << '`' << std::endl << std::endl;
        if(value->has_value()) {
          out << "default: " << '`' << *value << '`' << std::endl << std::endl;
        } else {
          out << "unset by default" << std::endl << std::endl;
        }

        out << "description: " << trim(std::string(func_description, (func_description == nullptr)? 0: strlen(func_description)));
        free((char*)func_description);
        out << std::endl << std::endl;


    }});

    //then print the remaining entries in the docs in a table
    bool first_option = true;
    std::set<std::string> skip_list;

    for_each_options(options, [&](const char* key, pressio_option* value) {
      if(std::string(key).find("/pressio:") == 0) {
        const std::string key_name = std::string(key).substr(strlen(name) + 2);

        if(first_option) {
          out << "### Options" << std::endl;
          first_option = false;
        }


        //get the description from the docs
        const char* func_description = nullptr;
        pressio_options_get_string(docs, key, &func_description);

        pressio_option_type type = pressio_option_get_type(value);

        out << "#### " << key_name << std::endl;
        out <<  std::endl;
        out << "type: `" << type << '`' << std::endl << std::endl;
        if(value->has_value()) {
          out << "default: " << '`' << *value << '`' << std::endl << std::endl;
        } else {
          out << "unset by default" << std::endl << std::endl;
        }
        //check configuration for enumerations
        const char** entries;
        size_t n_entries = 0;
        if(pressio_options_get_strings(configuration, key, &n_entries, &entries) == pressio_options_key_set) {
          skip_list.emplace(key);
          out << "options: ";
          for (int i = 0; i < n_entries; ++i) {
            out << entries[i];
            free((char*)entries[i]);
            if(i != n_entries -1) {
              out << ", ";
            }
          }
          free((char*)entries);
          out << std::endl << std::endl;
        }


        out << "description: " << trim(std::string(func_description, (func_description == nullptr)? 0: strlen(func_description)));
        out << std::endl;
        out << std::endl;
        free((char*)func_description);

      }
    });


    out << std::endl;
    bool first_config = true;

    for_each_options(configuration, [&](const char* key, pressio_option* value) {
      if(std::string(key).find("/pressio:") == 0) {
        const std::string key_name = std::string(key).substr(strlen(name) + 2);
        if(key_name == "pressio:thread_safe") {
          skip_list.emplace(key);
          return;
        }
        if(key_name == "pressio:stability") {
          skip_list.emplace(key);
          return;
        }

        if(skip_list.find(key) != skip_list.end()) {
          return;
        }

        if(first_config) {
          out << "### Configuration" << std::endl;
          first_config = false;
        }

        const char* func_description = nullptr;
        pressio_options_get_string(docs, key, &func_description);

        pressio_option_type type = pressio_option_get_type(value);

        out << "#### " << std::string(key).substr(strlen(name) + 2) << std::endl;
        out <<  std::endl;
        out << "type: `" << type << '`' << std::endl << std::endl;
        out << "value: " << '`' << *value << '`' << std::endl << std::endl;

        out << "description: " << trim(std::string(func_description, (func_description == nullptr)? 0: strlen(func_description)));
        out << std::endl;
        out << std::endl;

      }
    });

    if(skip_list.size()) {
      if(first_config) {
        out << "### Configuration" << std::endl;
        first_config = false;
      }

      out << "#### other configuration entries: " << std::endl;
      for (auto const& i : skip_list) {
        out << "+ " << i.substr(strlen(name) + 2) << std::endl;
      }
    }
    out << std::endl;
    out << std::endl;

    pressio_options_free(docs);
    pressio_options_free(options);
    pressio_options_free(configuration);
    pressio_options_free(metrics_results);
}

struct cmdline_args {
  bool generate_compressors = false;
  bool generate_io = false;
  bool generate_metrics = false;
  std::string outfile_path = "/proc/self/fd/0";
};

void usage() {
  std::cerr << R"(./generate_docs [flags]
  [flags]
    -m  [compressor,io,metrics]  mode -- the modes to include, default: none
  )" << std::endl;
}

cmdline_args parse_args(int argc, char* const argv[]) {
  cmdline_args args;

  int opt; while((opt = getopt(argc, argv, "hm:o:")) != -1) {
    switch(opt) {
      case 'm':
        switch (compat::string_view(optarg).at(0)) {
          case 'c':
            args.generate_compressors = true;
            break;
          case 'm':
            args.generate_metrics = true;
            break;
          case 'i':
            args.generate_io = true;
            break;
          default:
            std::cerr << "unexpected mode: " << optarg << std::endl;
            
            usage();
            exit(1);
        } 
        break;
      case 'o':
        args.outfile_path = optarg;
        break;
      case 'h':
        usage();
        exit(0);
      default:
        std::cerr << "unexpected flag: " << char(opt) << std::endl;
        usage();
        exit(1);
    }
  }

  return args;
}

void document_io(std::ostream& out, const char* metric, pressio_io const* c) {
    pressio_options* docs = pressio_io_get_documentation(c);
    pressio_options* options = pressio_io_get_options(c);
    pressio_options* configuration = pressio_io_get_configuration(c);

//    out << *configuration << std::endl;
//    out << *options << std::endl;
//    out << *docs << std::endl;

    const char* stability = nullptr;
    const char* description=nullptr;
    int32_t thread_safety = 0;
    pressio_options_get_string(configuration, "/pressio:pressio:stability", &stability);
    pressio_options_get_string(docs, "/pressio:pressio:description", &description);
    pressio_options_get_integer(configuration, "/pressio:pressio:thread_safe", &thread_safety);

    //print the id first and version information
    out << "## " << metric << std::endl;
    out << std::endl;
    out << "stability: " << std::string(stability, (stability == nullptr)? 0: strlen(stability)) << std::endl << std::endl;
    out << "thread_safety: " << pressio_thread_safety(thread_safety) << std::endl;
    out << std::endl;

    //then print the pressio:description entry
    out << trim(std::string(description, (description == nullptr)? 0: strlen(description))) << std::endl << std::endl;
    free((char*)description);
    free((char*)stability);

    //then print the remaining entries in the docs in a table
    bool first_option = true;
    std::set<std::string> skip_list;

    for_each_options(options, [&](const char* key, pressio_option* value) {
      if(std::string(key).find("/pressio:") == 0) {
        const std::string key_name = std::string(key).substr(strlen(name) + 2);

        if(first_option) {
          out << "### Options" << std::endl;
          first_option = false;
        }


        //get the description from the docs
        const char* func_description = nullptr;
        pressio_options_get_string(docs, key, &func_description);

        pressio_option_type type = pressio_option_get_type(value);

        out << "#### " << key_name << std::endl;
        out <<  std::endl;
        out << "type: `" << type << '`' << std::endl << std::endl;
        if(value->has_value()) {
          out << "default: " << '`' << *value << '`' << std::endl << std::endl;
        } else {
          out << "unset by default" << std::endl << std::endl;
        }
        //check configuration for enumerations
        const char** entries;
        size_t n_entries = 0;
        if(pressio_options_get_strings(configuration, key, &n_entries, &entries) == pressio_options_key_set) {
          skip_list.emplace(key);
          out << "options: ";
          for (int i = 0; i < n_entries; ++i) {
            out << entries[i];
            if(i != n_entries -1) {
              out << ", ";
            }
          }
          out << std::endl << std::endl;
        }


        out << "description: " << trim(std::string(func_description, (func_description == nullptr)? 0: strlen(func_description)));
        out << std::endl;
        out << std::endl;

        free((char*)func_description);
      }
    });


    out << std::endl;
    bool first_config = true;

    for_each_options(configuration, [&](const char* key, pressio_option* value) {
      if(std::string(key).find("/pressio:") == 0) {
        const std::string key_name = std::string(key).substr(strlen(name) + 2);
        if(key_name == "pressio:thread_safe") {
          skip_list.emplace(key);
          return;
        }
        if(key_name == "pressio:stability") {
          skip_list.emplace(key);
          return;
        }

        if(skip_list.find(key) != skip_list.end()) {
          return;
        }

        if(first_config) {
          out << "### Configuration" << std::endl;
          first_config = false;
        }

        const char* func_description = nullptr;
        pressio_options_get_string(docs, key, &func_description);

        pressio_option_type type = pressio_option_get_type(value);

        out << "#### " << std::string(key).substr(strlen(name) + 2) << std::endl;
        out <<  std::endl;
        out << "type: `" << type << '`' << std::endl << std::endl;
        out << "value: " << '`' << *value << '`' << std::endl << std::endl;

        out << "description: " << trim(std::string(func_description, (func_description == nullptr)? 0: strlen(func_description)));
        out << std::endl;
        out << std::endl;

        free((char*)func_description);

      }
    });

    if(skip_list.size()) {
      if(first_config) {
        out << "### Configuration" << std::endl;
        first_config = false;
      }

      out << "#### other configuration entries: " << std::endl;
      for (auto const& i : skip_list) {
        out << "+ " << i.substr(strlen(name) + 2) << std::endl;
      }
    }
    out << std::endl;
    out << std::endl;

    pressio_options_free(docs);
    pressio_options_free(options);
    pressio_options_free(configuration);
}



int main(int argc, char* const argv[])
{
  auto args = parse_args(argc, argv);
  auto* instance = pressio_instance();
  std::ofstream out(args.outfile_path);

  if(args.generate_compressors) {
    out << "# Compressors Modules {#compressors}" << std::endl;
    auto compressors = metas_list([]{return pressio_supported_compressors();});
    for (auto const& compressor : compressors) {
      pressio_compressor* c = pressio_get_compressor(instance, compressor.c_str());
      pressio_compressor_set_name(c, name);
      document_compressor(out, compressor, c);
      pressio_compressor_release(c);
    }
  }

  if(args.generate_metrics) {
    out << "# Metrics Modules {#metrics}" << std::endl;
    auto metrics = metas_list([]{return pressio_supported_metrics();});
    for (auto const& metric : metrics) {
      pressio_metrics* c = pressio_new_metric(instance, metric.c_str());
      pressio_metrics_set_name(c, name);
      document_metrics(out, metric.c_str(), c);
      pressio_metrics_free(c);
    }
  }

  if(args.generate_io) {
    out << "# IO Modules {#io}" << std::endl;
    auto metrics = metas_list([]{return pressio_supported_io_modules();});
    for (auto const& metric : metrics) {
      pressio_io* c = pressio_get_io(instance, metric.c_str());
      pressio_io_set_name(c, name);
      document_io(out, metric.c_str(), c);
      pressio_io_free(c);
    }
  }


  pressio_release(instance);
}
