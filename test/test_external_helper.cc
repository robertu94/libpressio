#include <string>
#include <vector>
#include <iostream>
#include <libpressio.h>
#include <libpressio_ext/io/posix.h>

struct cmdline_args {
  size_t api_version;
  std::string input;  
  std::string decompressed;  
  std::vector<size_t> dims;
  pressio_dtype type;
  std::string config_name;
  bool empty = true;
};

enum class cmdline {
  flag,
  api,
  input,
  decompressed,
  dim,
  type,
  config_name
};

cmdline_args parse_args(const int argc, const char* argv[])
{
  cmdline_args args;
  cmdline expected = cmdline::flag;

  //getopt long is a possible alternative for GPL software
  for (int i = 1; i < argc; ++i) {
    args.empty=false;
    std::string arg = argv[i];
    switch (expected) {
      case cmdline::flag:
        {
          if(arg == "--api") expected = cmdline::api;
          else if(arg == "--input") expected = cmdline::input;
          else if(arg == "--decompressed") expected = cmdline::decompressed;
          else if(arg == "--dim") expected = cmdline::dim;
          else if(arg == "--type") expected = cmdline::type;
          else if(arg == "--config_name") expected = cmdline::config_name;
          else {
            std::cerr << "Unexpected flag: " << arg << std::endl;
            exit(1);
          }
        }
        break;
      case cmdline::api:
        args.api_version = stoull(arg);
        expected = cmdline::flag;
        break;
      case cmdline::input:
        args.input = arg;
        expected = cmdline::flag;
        break;
      case cmdline::decompressed:
        args.decompressed = arg;
        expected = cmdline::flag;
        break;
      case cmdline::dim:
        args.dims.push_back(stoull(arg));
        expected = cmdline::flag;
        break;
      case cmdline::type:
        {
          if(arg == "float") args.type = pressio_float_dtype;
          else if(arg == "double") args.type = pressio_double_dtype;
          else if(arg == "int8") args.type = pressio_int8_dtype;
          else if(arg == "int16") args.type = pressio_int16_dtype;
          else if(arg == "int32") args.type = pressio_int32_dtype;
          else if(arg == "int64") args.type = pressio_int64_dtype;
          else if(arg == "uint8") args.type = pressio_uint8_dtype;
          else if(arg == "uint16") args.type = pressio_uint16_dtype;
          else if(arg == "uint32") args.type = pressio_uint32_dtype;
          else if(arg == "uint64") args.type = pressio_uint64_dtype;
          else {
            std::cerr << "unexpected type \"" << arg << '"' <<std::endl;
            exit(1);
          }
        }
        expected = cmdline::flag;
        break;
      case cmdline::config_name:
        args.config_name = arg;
        expected = cmdline::flag;
        break;
    }
  }

  return args;
}

int main(int argc, const char *argv[])
{
  std::cout << "external:api=3\n";
  std::cerr << std::endl;
  
  for (int i = 0; i < argc; ++i) {
    std::cerr << "argv[" << i << "] " << '\'' << argv[i] << '\'' << std::endl;
  }

  std::cerr << std::endl;

  auto args = parse_args(argc, argv);   
  if(!args.empty) {
    std::cout << "dims=" << args.dims.size() << '\n';

    auto input_buffer = pressio_data_new_owning(args.type, args.dims.size(), args.dims.data());
    auto input = pressio_io_data_path_read(input_buffer, args.input.c_str());
    if(input == nullptr) {
      std::cerr << "failed to read " << args.input << std::endl;
      exit(1);
    }

    auto decompressed_buffer = pressio_data_new_owning(args.type, args.dims.size(), args.dims.data());
    auto output = pressio_io_data_path_read(input_buffer, args.decompressed.c_str());
    if(output == nullptr) {
      std::cerr << "failed to read " << args.decompressed << std::endl;
      exit(1);
    }

    std::cerr << "testing warning" << std::endl;
    std::cout << "defaulted2=17.1" << std::endl;
    

    pressio_data_free(input);
    pressio_data_free(output);
    pressio_data_free(input_buffer);
    pressio_data_free(decompressed_buffer);
  } else {
    std::cout << "dims=0" << std::endl;
    std::cout << "defaulted=2.0" << std::endl;
    std::cout << "defaulted2=3.0" << std::endl;
  }

  std::flush(std::cout);
  return 0;
}
