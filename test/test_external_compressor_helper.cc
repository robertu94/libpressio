#include <getopt.h>
#include <string>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <libpressio_ext/cpp/pressio.h>
#include <libpressio_ext/cpp/options.h>
#include <libpressio_ext/cpp/dtype.h>
#include <libpressio_ext/cpp/printers.h>
#include <libpressio_ext/cpp/io.h>
#include <libpressio_ext/json/pressio_options_json.h>


/*
 * an implementation of the external compressor protocol
 */

using namespace std::string_literals;

enum mode {
    compress,
    decompress,
    get_options,
    set_options,
    get_configuration,
    get_documentation,
};
pressio_dtype parse_type(const char* str) {
    if(str == "float"s) return pressio_float_dtype;
    else if(str == "double"s) return pressio_double_dtype;
    else if(str == "byte"s) return pressio_byte_dtype;
    else throw std::runtime_error("unsupported type "s + str);
}

int main(int argc, char *argv[])
{
    std::cout << "external:api=5" << std::endl;
    try {
    enum mode mode;
    bool output_set = false;
    int api;
    std::string name;
    std::string config_name;
    std::string input, output;
    std::vector<size_t> idims, odims;
    pressio_dtype itype, otype;
    pressio_options options;
    pressio library;

    while(1) {
       int this_option_optind = optind ? optind : 1;
       int option_index = 0;
       static struct option long_options[] = {
           {"mode",     required_argument, 0,  'm' },
           {"api",     required_argument, 0,  'a' },
           {"input",     required_argument, 0,  'i' },
           {"idim",     required_argument, 0,  'd' },
           {"itype",     required_argument, 0,  't' },
           {"odim",     required_argument, 0,  'D' },
           {"otype",     required_argument, 0,  'T' },
           {"output",     required_argument, 0,  'I' },
           {"config_name",     required_argument, 0,  'c' },
           {"name",     required_argument, 0,  'n' },
           {"options",     required_argument, 0,  'o' },
           {0,         0,                 0,  0 }
       };

       int c = getopt_long(argc, argv, "",
                long_options, &option_index);
       if(c == -1) {
           break;
       }
       switch(c) {
           case 'm':
               if(optarg == "compress"s) { mode = mode::compress; }
               else if(optarg == "decompress"s) { mode = mode::decompress; }
               else if(optarg == "get_options"s) { mode = mode::get_options; }
               else if(optarg == "set_options"s) { mode = mode::set_options; }
               else if(optarg == "get_configuration"s) { mode = mode::get_configuration; }
               else if(optarg == "get_documentation"s) { mode = mode::get_documentation; }
               else {
                   std::cerr << "invalid mode " << mode << std::endl;
                   exit(1);
               }
               break;
            case 'a':
               api = atoi(optarg);
               break;
            case 'i':
               input = optarg;
               break;
            case 'd':
               idims.push_back(atoi(optarg));
               break;
            case 't':
               itype = parse_type(optarg);
               break;
            case 'I':
               output = optarg;
               break;
            case 'D':
               odims.push_back(atoi(optarg));
               output_set = true;
               break;
            case 'T':
               otype = parse_type(optarg);
               output_set = true;
               break;
            case 'c':
               config_name = optarg;
               break;
            case 'o':
               {
                   pressio_options* tmp = pressio_options_new_json(&library, optarg);
                   if(!tmp) {
                       std::cerr << library.err_msg() << std::endl;
                       exit(1);
                   }
                   options = std::move(*tmp);
                   pressio_options_free(tmp);
               }
               break;
            case 'n':
               name = optarg;
               break;
            default:
               std::cerr << "unexpected code" << c << std::endl;
       }
    }

    pressio_dtype compressed_type = pressio_double_dtype;
    options.cast("cast:cast", &compressed_type, pressio_conversion_special);

    switch(mode) {
        case mode::get_options:
            std::cout << "cast:cast=" << compressed_type  << std::endl;
        case mode::set_options:
            {
                std::string type;
                if(options.get("cast:cast", &type) == pressio_options_key_set) {
                    std::vector<std::string> types {
                        "float",
                        "double",
                        "int16",
                        "int8",
                    };
                    if(std::find(types.begin(), types.end(), type) == types.end()) {
                        std::cerr << "cast:cast must be one of float, double, int16, int8" << std::endl;
                        return 1;
                    }
                }
            }
            break;
        case mode::get_configuration:
            //allow these four types
            std::cout << "cast:cast=float"  << std::endl;
            std::cout << "cast:cast=double"  << std::endl;
            std::cout << "cast:cast=int16"  << std::endl;
            std::cout << "cast:cast=int8"  << std::endl;
            break;
        case mode::get_documentation:
            std::cout << "cast:cast=type to cast for the compressed sequence"  << std::endl;
            break;
        case mode::compress:
        case mode::decompress:
            {
                auto posix = library.get_io("posix");
                posix->set_options({
                    {"io:path",input}
                });
                pressio_data input_metadata = pressio_data::owning(itype, idims);
                pressio_data *input_data = posix->read(&input_metadata);
                if(!input_data) {
                    std::cerr << "failed to read input " << std::quoted(input) << std::endl;
                    exit(1);
                }

                double time;
                auto start = std::chrono::steady_clock::now();
                pressio_dtype output_type;
                pressio_data output;
                if(mode == mode::compress) {
                    output = input_data->cast(compressed_type);
                    output_type = compressed_type;
                    unlink("/tmp/out.data");
                    posix->set_options({
                        {"io:path", "/tmp/out.data"}
                    });
                    posix->write(&output);
                } else {
                    output = input_data->cast(otype);
                    output_type = otype;
                    unlink("/tmp/out.data");
                    posix->set_options({
                        {"io:path", "/tmp/out.data"}
                    });
                    posix->write(&output);
                }
                auto end = std::chrono::steady_clock::now();
                time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end-start).count();
                std::string output_type_str;
                switch(output_type) {
                    case pressio_float_dtype:
                        output_type_str = "float";
                        break;
                    case pressio_double_dtype:
                        output_type_str = "double";
                        break;
                    case pressio_int8_dtype:
                        output_type_str = "int8";
                        break;
                    case pressio_int16_dtype:
                        output_type_str = "int16";
                        break;
                    default:
                        std::cerr << "invalid output type" << std::endl;
                        exit(1);
                }

                std::cout << "output:0:path=" << "/tmp/out.data" << std::endl;
                std::cout << "output:0:dtype=" << output_type_str << std::endl;
                for (auto i : output.dimensions()) {
                    std::cout << "output:0:dims=" << i << std::endl;
                }
                std::cout << "metric:time=" << time << std::endl;
            }
            break;
    }

    } catch (std::exception const& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    
    return 0;
}
