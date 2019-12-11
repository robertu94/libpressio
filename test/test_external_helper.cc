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
};

enum class cmdline {
  flag,
  api,
  input,
  decompressed,
  dim,
  type
};

cmdline_args parse_args(const int argc, const char* argv[])
{
  cmdline_args args;
  cmdline expected = cmdline::flag;

  //getopt long is a possible alternative for GPL software
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    switch (expected) {
      case cmdline::flag:
        {
          if(arg == "--api") expected = cmdline::api;
          else if(arg == "--input") expected = cmdline::input;
          else if(arg == "--decompressed") expected = cmdline::decompressed;
          else if(arg == "--dim") expected = cmdline::dim;
          else if(arg == "--type") expected = cmdline::type;
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
    }
  }

  return args;
}

int main(int argc, const char *argv[])
{
  auto args = parse_args(argc, argv);   
  std::cout << "external:api=1\n";
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
  

  pressio_data_free(input);
  pressio_data_free(output);

  std::flush(std::cout);
  return 0;
}
