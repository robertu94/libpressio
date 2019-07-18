#include <optional>
#include <variant>
#include <string>
#include <map>
#include "lossy_options.h"
#include "lossy_option.h"

using option_type = std::variant<std::monostate, int, unsigned int, float, double,std::string,void*>;

namespace {
  template <class T>
  constexpr enum lossy_option_type lossy_type_to_enum = lossy_option_unset;
  template <>
  constexpr enum lossy_option_type lossy_type_to_enum<int> = lossy_option_int32_type;
  template <>
  constexpr enum lossy_option_type lossy_type_to_enum<unsigned int> = lossy_option_uint32_type;
  template <>
  constexpr enum lossy_option_type lossy_type_to_enum<float> = lossy_option_float_type;
  template <>
  constexpr enum lossy_option_type lossy_type_to_enum<double> = lossy_option_double_type;
  template <>
  constexpr enum lossy_option_type lossy_type_to_enum<std::string> = lossy_option_charptr_type;
  template <>
  constexpr enum lossy_option_type lossy_type_to_enum<const char*> = lossy_option_charptr_type;
  template <>
  constexpr enum lossy_option_type lossy_type_to_enum<void*> = lossy_option_userptr_type;
}

struct lossy_option {
  lossy_option()=default;
  lossy_option(option_type option): option(option) {}
  option_type option;
  std::optional<option_type> as(const enum lossy_option_type type, const enum lossy_conversion_safety safety = lossy_conversion_implicit) const;
  enum lossy_option_type type() const;
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
    using ValueType = std::remove_pointer_t<PointerType>;
    switch(key_status(key)){
      case lossy_options_key_set:
        {
          auto variant = get(key);
          if (std::holds_alternative<ValueType>(variant)) { 
            *value = std::get<ValueType>(variant);
            return lossy_options_key_set;
          } else {
            return lossy_options_key_exists;
          }
        }
        break;
      default:
        //value does not exist
        return lossy_options_key_does_not_exist;
    }
  }

  template <class PointerType>
  enum lossy_options_key_status cast(std::string const& key, PointerType value, enum lossy_conversion_safety safety) const {
    using ValueType = std::remove_pointer_t<PointerType>;
    switch(key_status(key)){
      case lossy_options_key_set:
        {
          auto variant = get(key);
          auto converted = lossy_option(variant).as(lossy_type_to_enum<ValueType>, safety);
          if(converted) {
            *value = std::get<ValueType>(*converted);
            return lossy_options_key_set;
          } else {
            return lossy_options_key_exists;
          }
        }
        break;
      default:
        //value does not exist
        return lossy_options_key_does_not_exist;
    }
  }

  auto begin() const {
    return std::begin(options);
  }
  auto end() const {
    return std::end(options);
  }

  auto begin() {
    return std::begin(options);
  }
  auto end() {
    return std::end(options);
  }

  using iterator = std::map<std::string, option_type>::iterator;
  using value_type = std::map<std::string, option_type>::value_type;

  iterator insert(iterator it, value_type const& value) {
    return options.insert(it, value);
  }


  private:
  std::map<std::string, option_type> options;
};

//special case for strings
template <>
enum lossy_options_key_status lossy_options::get(std::string const& key, const char** value) const;

template <>
enum lossy_options_key_status lossy_options::cast(std::string const& key, char** value, enum lossy_conversion_safety safety) const;
