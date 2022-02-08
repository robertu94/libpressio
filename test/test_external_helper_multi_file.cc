#include "libpressio_ext/cpp/data.h"
#include <string>
#include <vector>
#include <iostream>
#include <libpressio.h>
#include <libpressio_ext/io/posix.h>

struct cmdline_args {
  size_t api_version;
  std::string first_input;  
  std::string first_decompressed;  
  std::vector<size_t> first_dims;
  pressio_dtype first_type;
  std::string second_input;  
  std::string second_decompressed;  
  std::vector<size_t> second_dims;
  pressio_dtype second_type;
  std::string config_name;
  bool empty = true;
};

enum class cmdline {
  flag,
  api,
  first_input,
  first_decompressed,
  first_dim,
  first_type,
  second_input,
  second_decompressed,
  second_dim,
  second_type,
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
          else if(arg == "--first_input") expected = cmdline::first_input;
          else if(arg == "--first_decompressed") expected = cmdline::first_decompressed;
          else if(arg == "--first_dim") expected = cmdline::first_dim;
          else if(arg == "--first_type") expected = cmdline::first_type;
          else if(arg == "--second_input") expected = cmdline::second_input;
          else if(arg == "--second_decompressed") expected = cmdline::second_decompressed;
          else if(arg == "--second_dim") expected = cmdline::second_dim;
          else if(arg == "--second_type") expected = cmdline::second_type;
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
      case cmdline::second_input:
        args.second_input = arg;
        expected = cmdline::flag;
        break;
      case cmdline::second_decompressed:
        args.second_decompressed = arg;
        expected = cmdline::flag;
        break;
      case cmdline::second_dim:
        args.second_dims.push_back(stoull(arg));
        expected = cmdline::flag;
        break;
      case cmdline::second_type:
        {
          if(arg == "float") args.second_type = pressio_float_dtype;
          else if(arg == "double") args.second_type = pressio_double_dtype;
          else if(arg == "int8") args.second_type = pressio_int8_dtype;
          else if(arg == "int16") args.second_type = pressio_int16_dtype;
          else if(arg == "int32") args.second_type = pressio_int32_dtype;
          else if(arg == "int64") args.second_type = pressio_int64_dtype;
          else if(arg == "uint8") args.second_type = pressio_uint8_dtype;
          else if(arg == "uint16") args.second_type = pressio_uint16_dtype;
          else if(arg == "uint32") args.second_type = pressio_uint32_dtype;
          else if(arg == "uint64") args.second_type = pressio_uint64_dtype;
          else {
            std::cerr << "unexpected type \"" << arg << '"' <<std::endl;
            exit(1);
          }
        }
        expected = cmdline::flag;
        break;
      case cmdline::first_input:
        args.first_input = arg;
        expected = cmdline::flag;
        break;
      case cmdline::first_decompressed:
        args.first_decompressed = arg;
        expected = cmdline::flag;
        break;
      case cmdline::first_dim:
        args.first_dims.push_back(stoull(arg));
        expected = cmdline::flag;
        break;
      case cmdline::first_type:
        {
          if(arg == "float") args.first_type = pressio_float_dtype;
          else if(arg == "double") args.first_type = pressio_double_dtype;
          else if(arg == "int8") args.first_type = pressio_int8_dtype;
          else if(arg == "int16") args.first_type = pressio_int16_dtype;
          else if(arg == "int32") args.first_type = pressio_int32_dtype;
          else if(arg == "int64") args.first_type = pressio_int64_dtype;
          else if(arg == "uint8") args.first_type = pressio_uint8_dtype;
          else if(arg == "uint16") args.first_type = pressio_uint16_dtype;
          else if(arg == "uint32") args.first_type = pressio_uint32_dtype;
          else if(arg == "uint64") args.first_type = pressio_uint64_dtype;
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

void check_buffer_contents(pressio_data* data, int32_t value) {
  auto ptr = static_cast<int32_t*>(pressio_data_ptr(data, nullptr));
  if(ptr[0] != value) {
    std::cerr << "expected " << value << "but got "  << ptr[0] <<  std::endl;
    exit(2);
  }
}

int main(int argc, const char *argv[])
{
  std::cout << "external:api=3\n";
  std::cerr << std::endl;
  std::cerr << "multi-file" << std::endl;
  
  for (int i = 0; i < argc; ++i) {
    std::cerr << "argv[" << i << "] " << '\'' << argv[i] << '\'' << std::endl;
  }

  std::cerr << std::endl;

  auto args = parse_args(argc, argv);   
  if(!args.empty) {
    std::cout << "first_dims=" << args.first_dims.size() << '\n';
    std::cout << "second_dims=" << args.second_dims.size() << '\n';

    auto first_input_buffer = pressio_data_new_owning(args.first_type, args.first_dims.size(), args.first_dims.data());
    auto first_input = pressio_io_data_path_read(first_input_buffer, args.first_input.c_str());
    if(first_input == nullptr || pressio_data_ptr(first_input, nullptr) == nullptr) {
      std::cerr << "failed to read first_in " << args.first_input << std::endl;
      exit(1);
    }
    check_buffer_contents(first_input, 1);

    auto first_decompressed_buffer = pressio_data_new_owning(args.first_type, args.first_dims.size(), args.first_dims.data());
    auto first_output = pressio_io_data_path_read(first_decompressed_buffer, args.first_decompressed.c_str());
    if(first_output == nullptr || pressio_data_ptr(first_output, nullptr) == nullptr) {
      std::cerr << "failed to read first_out " << args.first_decompressed << std::endl;
      exit(1);
    }
    check_buffer_contents(first_output, 1);

    auto second_input_buffer = pressio_data_new_owning(args.second_type, args.second_dims.size(), args.second_dims.data());
    auto second_input = pressio_io_data_path_read(second_input_buffer, args.second_input.c_str());
    if(second_input == nullptr || pressio_data_ptr(second_input, nullptr) == nullptr) {
      std::cerr << "failed to read second_in " << args.second_input << std::endl;
      exit(1);
    }
    check_buffer_contents(second_input, 2);

    auto second_decompressed_buffer = pressio_data_new_owning(args.second_type, args.second_dims.size(), args.second_dims.data());
    auto second_output = pressio_io_data_path_read(second_decompressed_buffer, args.second_decompressed.c_str());
    if(second_output == nullptr || pressio_data_ptr(second_output, nullptr) == nullptr) {
      std::cerr << "failed to read second_out " << args.second_decompressed << std::endl;
      exit(1);
    }
    check_buffer_contents(second_output, 2);

    std::cerr << "testing warning" << std::endl;
    std::cout << "defaulted2=17.1" << std::endl;
    

    pressio_data_free(first_decompressed_buffer);
    pressio_data_free(second_decompressed_buffer);
    pressio_data_free(second_input_buffer);
    pressio_data_free(first_input_buffer);
    pressio_data_free(first_input);
    pressio_data_free(first_output);
    pressio_data_free(second_input);
    pressio_data_free(second_output);
  } else {
    std::cout << "first_dims=0" << std::endl;
    std::cout << "second_dims=0" << std::endl;
    std::cout << "defaulted=2.0" << std::endl;
    std::cout << "defaulted2=3.0" << std::endl;
  }

  std::flush(std::cout);
  return 0;
}
