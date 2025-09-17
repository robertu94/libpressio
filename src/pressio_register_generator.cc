#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <unistd.h>

struct cli_args {
    std::string config;
    std::string output_path;
};
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out, cli_args const& args) {
    return out << "{.path=" << std::quoted(args.output_path) << ", .config=" << std::quoted(args.config.substr(0,50) + "...") << "}";
}

struct plugin {
    std::string type;
    std::string name;
    std::string path;
};
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out, plugin const& args) {
    return out << "{.path=" << std::quoted(args.path) << ", .name=" << std::quoted(args.name) << ", .type=" << std::quoted(args.type) << "}";
}

int main(int argc, char* argv[]) {
    cli_args args;
    int opt = 0;
    while((opt = getopt(argc, argv, "o:c:")) != -1) {
        switch(opt) {
            case 'o':
                args.output_path = optarg;
                break;
            case 'c':
                args.config = optarg;
                break;
            default:
                std::cerr << "unknown argument \"" << opt << "\"" <<  std::endl;
                return 1;
        }
    }
    std::cerr << args << std::endl;


    std::vector<plugin> plugins;
    std::stringstream ss(args.config);
    std::string item; 
    while(std::getline(ss, item, ';')) {
        std::string name,type;
        auto extension_pos = item.find_last_of(".");
        auto suffix_pos = item.find_last_of("/");
        auto parent_pos = item.find_last_of("/", suffix_pos - 1);
        if (extension_pos == std::string::npos || suffix_pos == std::string::npos ||
            parent_pos == std::string::npos) {
          std::cout << "unable to parse path " << std::quoted(item);
          return 1;
        }
        name = item.substr(suffix_pos+1, extension_pos - suffix_pos - 1);
        type = item.substr(parent_pos+1, suffix_pos - parent_pos - 1);
        plugins.emplace_back(plugin{type, name, item});
        std::cerr << plugins.back() << std::endl;
    }

    std::sort(plugins.begin(), plugins.end(), [](plugin const& lhs, plugin const& rhs) { 
            if(lhs.type == rhs.type) return lhs.name < rhs.name;
            else return lhs.type < rhs.type;
        });
    
    std::string header = R"(
    #include <iostream>
    #include "libpressio_ext/cpp/registry.h"

    
    )";
    std::stringstream fwd_decl;
    std::stringstream body;
    body << R"(extern "C" void pressio_register_all() {)" <<std::endl;
    fwd_decl << "namespace libpressio {" << std::endl;;
    std::string last_type = plugins.front().type;
    fwd_decl << "\tnamespace " << last_type << " {" << std::endl;
    for(auto const& p: plugins) {
        if(last_type != p.type) {
            //starting the nex type
            fwd_decl << "\t} /* namespace " << last_type << "*/" << std::endl;
            fwd_decl << "\tnamespace " << p.type << " {" << std::endl;
            last_type = p.type;
        }
        fwd_decl << "\t\tnamespace " << p.name << "_ns {" << std::endl;
        fwd_decl << "\t\t\textern pressio_register registration;" << std::endl;
        fwd_decl << "\t\t}" << std::endl;

        body << "libpressio::" << p.type << "::" << p.name << "_ns::registration.ensure_registered();" << std::endl;;
    }
    fwd_decl << "\t} /*namespace " << last_type << "*/" << std::endl;
    fwd_decl << "} /*namespace libpressio*/" << std::endl;

    std::string footer = R"(
    }
    )";

    std::cerr << "generating with " << plugins.size() << std::endl;
    std::fstream out(args.output_path, std::ios::out|std::ios::trunc);
    out << header << fwd_decl.str() << body.str() << footer << std::endl;
    return 0;
}
