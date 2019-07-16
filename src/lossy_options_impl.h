#include <variant>
#include <string>
#include <map>
#include "lossy_options.h"

using option_type = std::variant<std::monostate, int, unsigned int, float, double,std::string,void*>;

struct lossy_option {
  lossy_option()=default;
  lossy_option(option_type option): option(option) {}
  option_type option;
};

struct lossy_options{
  lossy_options_key_status key_status(std::string const& key) const {
    auto it = options.find(key);
    if(it == options.end()) {
      return lossy_options_key_does_not_exist;
    } else if (std::holds_alternative<std::monostate>(it->second)) {
      return lossy_options_key_exists;
    } else { 
      return lossy_options_key_set;
    }
  }

  void set(std::string const& key,  option_type const& value) {
    options[key] = value;
  }

  option_type const& get(std::string const& key) const {
    return options.at(key);
  }

  template <class PointerType>
  enum lossy_options_key_status get(std::string const& key, PointerType value) const {
    switch(key_status(key)){
      case lossy_options_key_set:
        {
        auto opt = std::get_if<std::remove_pointer_t<PointerType>>((&get(key)));
        if(opt != nullptr) {
          *value = *opt;
          return lossy_options_key_set;
        } else { 
          //type does not match
          return lossy_options_key_exists;
        }
        }
      default:
        //value does not exist
        return lossy_options_key_does_not_exist;
    }
  }
  

  auto begin() {
    return std::begin(options);
  }
  auto end() {
    return std::end(options);
  }

  private:
  std::map<std::string, option_type> options;
};

//special case for strings
template <>
enum lossy_options_key_status lossy_options::get(std::string const& key, const char** value) const;
