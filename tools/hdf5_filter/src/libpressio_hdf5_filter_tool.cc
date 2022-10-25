#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <unistd.h>
#include <libpressio_hdf5_filter_impl.h>
#include <libpressio_hdf5_filter.h>
#include <libpressio_ext/cpp/libpressio.h>

pressio_dtype parse_dtype(std::string const& arg) {
  if(arg == "int8") return pressio_int8_dtype;
  else if(arg == "int16") return pressio_int16_dtype;
  else if(arg == "int32") return pressio_int32_dtype;
  else if(arg == "int64") return pressio_int64_dtype;
  else if(arg == "uint8") return pressio_uint8_dtype;
  else if(arg == "byte") return pressio_byte_dtype;
  else if(arg == "uint16") return pressio_uint16_dtype;
  else if(arg == "uint32") return pressio_uint32_dtype;
  else if(arg == "uint64") return pressio_uint64_dtype;
  else if(arg == "float") return pressio_float_dtype;
  else if(arg == "double") return pressio_double_dtype;
  else if(arg == "bool") return pressio_bool_dtype;
  else {
    throw std::runtime_error("unsupported dtype: " + arg);
  }
}
std::pair<std::string, std::string> parse_option(std::string const& arg) {
  auto eq_pos = arg.find('=');
  if(eq_pos != std::string::npos) {
    auto name = arg.substr(0, eq_pos);
    auto rest = arg.substr(eq_pos+1);
    return {name, rest};
  } else {
    std::stringstream ss;
    ss << "invalid option " << std::quoted(arg);
    throw std::runtime_error(ss.str());
  }

}

pressio_options early_options(std::map<std::string, std::vector<std::string>> const& early) {
  pressio_options opts;
  for (auto const& i : early) {
    if(i.second.size() == 1) opts.set(i.first, i.second.front());
    else opts.set(i.first, i.second);
  }
  return opts;
}
pressio_options late_options(std::map<std::string, std::vector<std::string>> const& early, pressio_options&& opts) {
  pressio_option opt;
  pressio_options ret;
  for (auto const& i : early) {
    std::string const& name = i.first;
    if(i.second.size() == 1) opt = i.second.front();
    else opt = i.second;

    if(opts.key_status(name) != pressio_options_key_does_not_exist) {
      ret.set(name, opts.get(name));
      ret.cast_set(name, opt, pressio_conversion_special);
    }
  }
  return ret;
}
void emplace(std::map<std::string, std::vector<std::string>>& map, std::pair<std::string, std::string> && opt) {
  map[opt.first].emplace_back(opt.second);
}

int main(int argc, char *argv[])
{
  try {
    bool verbose = false;
    std::map<std::string, std::vector<std::string>> early;
    std::map<std::string, std::vector<std::string>> late;
    compression_options options;
    int opt;
    while((opt = getopt(argc, argv, "d:t:b:o:v")) != -1) {
      switch(opt) {
        case 'd':
          options.dims.emplace_back(std::stoull(optarg));
          break;
        case 't':
          options.dtype = parse_dtype(optarg);
          break;
        case 'b':
          emplace(early, parse_option(optarg));
          break;
        case 'o':
          emplace(late,parse_option(optarg));
          break;
        case 'v':
          verbose = true;
          break;
      }
    }
    
    if(optind >= argc) {
     options.compressor_id = "pressio";
    } else {
     options.compressor_id = argv[optind];
    }

    pressio library;
    pressio_compressor c = library.get_compressor(options.compressor_id);
    c->set_options(early_options(early));
    c->set_options(late_options(late, c->get_options()));
    options.options = c->get_options();
    if(verbose) {
      std::cerr << options.options << std::endl;
    }
    auto cd_vals = get_cd_values_from_options(options);

    std::cout << "UD=";
    std::cout << H5Z_FILTER_LIBPRESSIO;
    for (size_t i = 0; i < cd_vals.size(); ++i) {
      std::cout << cd_vals[i];
      if(i != (cd_vals.size() - 1)) {
        std::cout << ',';
      }
    }
    std::cout << std::endl;

  } catch(std::runtime_error const& ex) {
     std::cerr << ex.what() << std::endl;
  }
  
  return 0;
}
