#include <variant>
#include <map>
#include <algorithm>
#include <iterator>
#include <string>
#include <cstring>
#include "lossy_options_impl.h"


void lossy_options_free(struct lossy_options* options) {
  delete options;
}

struct lossy_options* lossy_options_copy(struct lossy_options const* options) {
  return new lossy_options(*options);
}

struct lossy_options* lossy_options_new() {
  return new lossy_options;
}

void lossy_options_clear(struct lossy_options* options, const char* key) {
  options->set(key, std::monostate());
}
#define lossy_options_define_type_impl_get(name, type) \
  void lossy_options_set_##name(struct lossy_options* options, const char* key, type value) { \
    options->set(key, value); \
  }
#define lossy_options_define_type_impl_set(name, type) \
  enum lossy_options_key_status lossy_options_get_##name(struct lossy_options const* options, const char* key, type* value) { \
    return options->get(key, value); \
  }
#define lossy_options_define_type_impl_cast(name, type) \
  enum lossy_options_key_status lossy_options_cast_##name(struct lossy_options const* options, const char* key, const enum lossy_conversion_safety safety, type* value) { \
    return options->cast(key, value, safety); \
  }
#define lossy_options_define_type_impl_as(name, type) \
  enum lossy_options_key_status lossy_options_as_##name(struct lossy_options const* options, const char* key, type* value) { \
    return options->cast(key, value, lossy_conversion_implicit); \
  } 

#define lossy_options_define_type_impl(name, type) \
  lossy_options_define_type_impl_get(name, type)  \
  lossy_options_define_type_impl_set(name, type)  \
  lossy_options_define_type_impl_cast(name, type)  \
  lossy_options_define_type_impl_as(name, type) 

lossy_options_define_type_impl(uinteger, unsigned int)
lossy_options_define_type_impl(integer, int)
lossy_options_define_type_impl(float, float)
lossy_options_define_type_impl(double, double)
lossy_options_define_type_impl(userptr, void*)

lossy_options_define_type_impl_get(string, const char*)
lossy_options_define_type_impl_set(string, const char*)
lossy_options_define_type_impl_cast(string, char*)
lossy_options_define_type_impl_as(string, char*)


lossy_options_key_status lossy_options_exists(struct lossy_options const* options, const char* key) {
  return options->key_status(key);
}

lossy_option* lossy_options_get(struct lossy_options const* options, const char* key) {
  return new lossy_option(options->get(key));
}

struct lossy_options* lossy_options_merge(struct lossy_options const* lhs, struct lossy_options const* rhs) {
  struct lossy_options* merged = new lossy_options(*lhs);
  std::copy(std::begin(*rhs), std::end(*rhs), std::inserter(*merged, merged->begin()));
  return merged;
}


//special case for strings
template <>
enum lossy_options_key_status lossy_options::get(std::string const& key, const char** value) const {
  switch(key_status(key)){
    case lossy_options_key_set:
      {
        const std::string* opt = std::get_if<std::string>((&get(key)));
        if(opt!=nullptr) {
          *value = opt->c_str();
          return lossy_options_key_set;
        }  else { 
          return lossy_options_key_exists;
        }
      }
    default:
      return lossy_options_key_does_not_exist;
  }
}



template <>
enum lossy_options_key_status lossy_options::cast(std::string const& key, char** value, enum lossy_conversion_safety safety) const {
  using ValueType = std::string;
  switch(key_status(key)){
    case lossy_options_key_set:
      {
        auto variant = get(key);
        auto converted = lossy_option(variant).as(lossy_type_to_enum<ValueType>, safety);
        if(converted) {
          *value = strdup(std::get<ValueType>(*converted).c_str());
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
