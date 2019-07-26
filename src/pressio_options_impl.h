#include <optional>
#include <variant>
#include <string>
#include <map>
#include "pressio_options.h"
#include "pressio_option.h"


namespace {

  template <class T>
  enum pressio_option_type pressio_type_to_enum;
  template <>
  constexpr enum pressio_option_type pressio_type_to_enum<int> = pressio_option_int32_type;
  template <>
  constexpr enum pressio_option_type pressio_type_to_enum<unsigned int> = pressio_option_uint32_type;
  template <>
  constexpr enum pressio_option_type pressio_type_to_enum<float> = pressio_option_float_type;
  template <>
  constexpr enum pressio_option_type pressio_type_to_enum<double> = pressio_option_double_type;
  template <>
  constexpr enum pressio_option_type pressio_type_to_enum<std::string> = pressio_option_charptr_type;
  template <>
  constexpr enum pressio_option_type pressio_type_to_enum<const char*> = pressio_option_charptr_type;
  template <>
  constexpr enum pressio_option_type pressio_type_to_enum<void*> = pressio_option_userptr_type;
}

using option_type = std::variant<std::monostate,
      std::optional<int>,
      std::optional<unsigned int>,
      std::optional<float>,
      std::optional<double>,
      std::optional<std::string>,
      std::optional<void*>
      >;

struct pressio_option {
  pressio_option()=default;

  template<class T>
  pressio_option(T option): option(std::optional<T>(option)) {}

  template<>
  pressio_option(std::monostate option): option(option) {}

  pressio_option as(const enum pressio_option_type type, const enum pressio_conversion_safety safety = pressio_conversion_implicit) const;
  enum pressio_option_type type() const;

  template <class T>
  constexpr bool holds_alternative() const {
    return std::holds_alternative<std::optional<T>>(option);
  }

  template <>
  constexpr bool holds_alternative<std::monostate>() const;

  template <class T>
  constexpr std::optional<T> const& get() const{
    return std::get<std::optional<T>>(option);
  }

  template <class T>
  constexpr T const& get_value() const{
    return get<T>().value();
  }

  bool has_value() const {
    if (holds_alternative<std::monostate>()) return false;
    else {
      switch(type())
      {
        case pressio_option_int32_type:
          return get<int>().has_value();
        case pressio_option_uint32_type:
          return get<unsigned int>().has_value();
        case pressio_option_float_type:
          return get<float>().has_value();
        case pressio_option_double_type:
          return get<double>().has_value();
        case pressio_option_charptr_type:
          return get<std::string>().has_value();
        case pressio_option_userptr_type:
          return get<void*>().has_value();
        case pressio_option_unset_type:
          return false;
      }
    }
  }

  template <class T>
  void set(T v) {
    option = std::optional(v);
  }

  void set_type(pressio_option_type type) {
    switch(type)
    {
      case pressio_option_charptr_type:
        option = std::optional<std::string>();
        break;
      case pressio_option_userptr_type:
        option = std::optional<void*>();
        break;
      case pressio_option_int32_type:
        option = std::optional<int>();
        break;
      case pressio_option_uint32_type:
        option = std::optional<int>();
        break;
      case pressio_option_float_type:
        option = std::optional<float>();
        break;
      case pressio_option_double_type:
        option = std::optional<double>();
        break;
      case pressio_option_unset_type:
        option = std::monostate{};
        break;
    }
  }
  enum pressio_options_key_status cast_set(struct pressio_option const& rhs, enum pressio_conversion_safety safety = pressio_conversion_implicit) { 
    auto casted = rhs.as(type(), safety);
    if (casted.has_value()) {
      *this = casted;
      return pressio_options_key_set;
    } else {
      return pressio_options_key_exists; 
    }
  }

  private:
  option_type option;
};

template <>
constexpr bool pressio_option::holds_alternative<std::monostate>() const {
  return std::holds_alternative<std::monostate>(option);
}

struct pressio_options{

  pressio_options_key_status key_status(std::string const& key) const {
    auto it = options.find(key);
    if(it == options.end()) {
      return pressio_options_key_does_not_exist;
    } else if (it->second.has_value()) {
      return pressio_options_key_set;
    } else { 
      return pressio_options_key_exists;
    }
  }

  void set(std::string const& key,  pressio_option const& value) {
    options[key] = value;
  }

  enum pressio_options_key_status cast_set(std::string const& key,  pressio_option const& value, enum pressio_conversion_safety safety= pressio_conversion_implicit) {
    switch(key_status(key))
    {
      case pressio_options_key_set:
      case pressio_options_key_exists:
        return options[key].cast_set(value, safety);
      default:
        return pressio_options_key_does_not_exist;
    }

  }

  void set_type(std::string const& key, pressio_option_type type) {
    options[key].set_type(type);
  }

  pressio_option const& get(std::string const& key) const {
    return options.at(key);
  }

  template <class PointerType>
  enum pressio_options_key_status get(std::string const& key, PointerType value) const {
    using ValueType = std::remove_pointer_t<PointerType>;
    switch(key_status(key)){
      case pressio_options_key_set:
        {
          auto variant = get(key);
          if (variant.holds_alternative<ValueType>()) { 
            *value = variant.get_value<ValueType>();
            return pressio_options_key_set;
          } else {
            return pressio_options_key_exists;
          }
        }
        break;
      default:
        //value does not exist
        return pressio_options_key_does_not_exist;
    }
  }

  template <class PointerType>
  enum pressio_options_key_status cast(std::string const& key, PointerType value, enum pressio_conversion_safety safety) const {
    using ValueType = std::remove_pointer_t<PointerType>;
    switch(key_status(key)){
      case pressio_options_key_set:
        {
          auto variant = get(key);
          auto converted = pressio_option(variant).as(pressio_type_to_enum<ValueType>, safety);
          if(converted.has_value()) {
            *value = converted.template get_value<ValueType>();
            return pressio_options_key_set;
          } else {
            return pressio_options_key_exists;
          }
        }
        break;
      default:
        //value does not exist
        return pressio_options_key_does_not_exist;
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

  using iterator = std::map<std::string, pressio_option>::iterator;
  using value_type = std::map<std::string, pressio_option>::value_type;

  iterator insert(iterator it, value_type const& value) {
    return options.insert(it, value);
  }


  private:
  std::map<std::string, pressio_option> options;
};

//special case for strings
template <>
enum pressio_options_key_status pressio_options::get(std::string const& key, const char** value) const;

template <>
enum pressio_options_key_status pressio_options::cast(std::string const& key, char** value, enum pressio_conversion_safety safety) const;
