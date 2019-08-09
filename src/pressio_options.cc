#include <variant>
#include <map>
#include <algorithm>
#include <iterator>
#include <string>
#include <cstring>
#include "pressio_options_impl.h"


void pressio_options_free(struct pressio_options* options) {
  delete options;
}

struct pressio_options* pressio_options_copy(struct pressio_options const* options) {
  return new pressio_options(*options);
}

struct pressio_options* pressio_options_new() {
  return new pressio_options;
}

void pressio_options_clear(struct pressio_options* options, const char* key) {
  options->set(key, std::monostate());
}

void pressio_options_set(struct pressio_options* options, const char* key, struct pressio_option* option) {
  options->set(key, *option);
}

void pressio_options_set_type(struct pressio_options* options, const char* key, pressio_option_type type) {
  options->set_type(key, type);
}

enum pressio_options_key_status pressio_options_cast_set(struct pressio_options* options, const char* key, struct pressio_option* option, enum pressio_conversion_safety safety) {
  return options->cast_set(key, *option, safety);
}

enum pressio_options_key_status pressio_options_as_set(struct pressio_options* options, const char* key, struct pressio_option* option) {
  return options->cast_set(key, *option);
}


#define pressio_options_define_type_impl_get(name, type) \
  void pressio_options_set_##name(struct pressio_options* options, const char* key, type value) { \
    options->set(key, value); \
  }
#define pressio_options_define_type_impl_set(name, type) \
  enum pressio_options_key_status pressio_options_get_##name(struct pressio_options const* options, const char* key, type* value) { \
    return options->get(key, value); \
  }
#define pressio_options_define_type_impl_cast(name, type) \
  enum pressio_options_key_status pressio_options_cast_##name(struct pressio_options const* options, const char* key, const enum pressio_conversion_safety safety, type* value) { \
    return options->cast(key, value, safety); \
  }
#define pressio_options_define_type_impl_as(name, type) \
  enum pressio_options_key_status pressio_options_as_##name(struct pressio_options const* options, const char* key, type* value) { \
    return options->cast(key, value, pressio_conversion_implicit); \
  } 

#define pressio_options_define_type_impl(name, type) \
  pressio_options_define_type_impl_get(name, type)  \
  pressio_options_define_type_impl_set(name, type)  \
  pressio_options_define_type_impl_cast(name, type)  \
  pressio_options_define_type_impl_as(name, type) 

pressio_options_define_type_impl(uinteger, unsigned int)
pressio_options_define_type_impl(integer, int)
pressio_options_define_type_impl(float, float)
pressio_options_define_type_impl(double, double)
pressio_options_define_type_impl(userptr, void*)

pressio_options_define_type_impl_get(string, const char*)
pressio_options_define_type_impl_set(string, const char*)
pressio_options_define_type_impl_cast(string, char*)
pressio_options_define_type_impl_as(string, char*)


pressio_options_key_status pressio_options_exists(struct pressio_options const* options, const char* key) {
  return options->key_status(key);
}

pressio_option* pressio_options_get(struct pressio_options const* options, const char* key) {
  return new pressio_option(options->get(key));
}

struct pressio_options* pressio_options_merge(struct pressio_options const* lhs, struct pressio_options const* rhs) {
  struct pressio_options* merged = new pressio_options(*lhs);
  std::copy(std::begin(*rhs), std::end(*rhs), std::inserter(*merged, merged->begin()));
  return merged;
}


//special case for strings
template <>
enum pressio_options_key_status pressio_options::get(std::string const& key, const char** value) const {
  switch(key_status(key)){
    case pressio_options_key_set:
      {
        auto opt = get(key);
        if(opt.holds_alternative<std::string>()) {
          *value = opt.get_value<std::string>().c_str();
          return pressio_options_key_set;
        }  else { 
          return pressio_options_key_exists;
        }
      }
    default:
      return pressio_options_key_does_not_exist;
  }
}



template <>
enum pressio_options_key_status pressio_options::cast(std::string const& key, char** value, enum pressio_conversion_safety safety) const {
  using ValueType = std::string;
  switch(key_status(key)){
    case pressio_options_key_set:
      {
        auto variant = get(key);
        auto converted = pressio_option(variant).as(pressio_type_to_enum<ValueType>, safety);
        if(converted.has_value()) {
          *value = strdup(converted.get_value<std::string>().c_str());
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
