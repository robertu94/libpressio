#include <map>
#include <algorithm>
#include <string>
#include <cstring>
#include <sstream>
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "std_compat/std_compat.h"


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
  options->set(key, compat::monostate());
}

void pressio_options_set(struct pressio_options* options, const char* key, struct pressio_option* option) {
  options->set(key, *option);
}

void pressio_options_set_type(struct pressio_options* options, const char* key, pressio_option_type type) {
  options->set_type(key, type);
}

enum pressio_options_key_status pressio_options_cast_set(struct pressio_options* options, const char* key, struct pressio_option const* option, enum pressio_conversion_safety safety) {
  return options->cast_set(key, *option, safety);
}

enum pressio_options_key_status pressio_options_as_set(struct pressio_options* options, const char* key, struct pressio_option* option) {
  return options->cast_set(key, *option);
}

size_t pressio_options_size(struct pressio_options const* options) {
  return options->size();
}

size_t pressio_options_num_set(struct pressio_options const* options) {
  return options->num_set();
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

pressio_options_define_type_impl(bool, bool)
pressio_options_define_type_impl(uinteger8, uint8_t)
pressio_options_define_type_impl(integer8, int8_t)
pressio_options_define_type_impl(uinteger16, uint16_t)
pressio_options_define_type_impl(integer16, int16_t)
pressio_options_define_type_impl(uinteger64, uint64_t)
pressio_options_define_type_impl(integer64, int64_t)
pressio_options_define_type_impl(uinteger, uint32_t)
pressio_options_define_type_impl(integer, int32_t)
pressio_options_define_type_impl(float, float)
pressio_options_define_type_impl(double, double)
pressio_options_define_type_impl(userptr, void*)

//special case: string -- for memory management
void pressio_options_set_string(struct pressio_options* options, const char* key, const char* value) { \
  std::string value_tmp = value;
  options->set(key, value_tmp);
}
enum pressio_options_key_status pressio_options_get_string(struct pressio_options const* options, const char* key, const char** value) { \
  std::string value_tmp;
  auto status = options->get(key, &value_tmp);
  if(status == pressio_options_key_set) {
    *value = strndup(value_tmp.c_str(), value_tmp.size());
  }
  return status;
}

enum pressio_options_key_status pressio_options_cast_string(struct pressio_options const* options, const char* key, const enum pressio_conversion_safety safety, char** value) {
  std::string value_tmp;
  auto status = options->cast(key, &value_tmp, safety); 
  if(status == pressio_options_key_set) {
    *value = strndup(value_tmp.c_str(), value_tmp.size());
  }
  return status;
}
enum pressio_options_key_status pressio_options_as_string(struct pressio_options const* options, const char* key, char** value) {
  return pressio_options_cast_string(options, key, pressio_conversion_implicit, value);
} 
//special case: data -- to dereference the pointer
void pressio_options_set_data(struct pressio_options* options, const char* key, struct pressio_data* value) {
  options->set(key, *value);
}
enum pressio_options_key_status pressio_options_get_data(struct pressio_options const* options, const char* key, struct pressio_data** value) { \
  return options->get(key, *value);
}

enum pressio_options_key_status pressio_options_cast_data(struct pressio_options const* options, const char* key, const enum pressio_conversion_safety safety, struct pressio_data** value) {
  return options->cast(key, *value, safety);
}
enum pressio_options_key_status pressio_options_as_data(struct pressio_options const* options, const char* key, struct pressio_data** value) {
  return pressio_options_cast_data(options, key, pressio_conversion_implicit, value);
} 

//special case: strings -- to get/pass length information
void pressio_options_set_strings(struct pressio_options* options, const char* key, size_t size, const char* const* values) {
  std::vector<std::string> strings(values, values+size);
  return options->set(key, strings);
}
enum pressio_options_key_status pressio_options_get_strings(struct pressio_options const* options, const char* key, size_t* size, const char***  values) {
  std::vector<std::string> strings;
  auto status = options->get(key, &strings);
    if(status == pressio_options_key_set) {
    *size = strings.size();
    *values = static_cast<const char**>(malloc(sizeof(const char*)**size));
    for (size_t i = 0; i < *size; ++i) {
      (*values)[i] = strndup(strings[i].c_str(), strings[i].size());
    }
  } else {
    *size = 0;
  }
  return status;
}
enum pressio_options_key_status pressio_options_cast_strings(struct pressio_options
      const* options, const char* key, const enum pressio_conversion_safety safety,
      size_t* size, char*** values) {
  std::vector<std::string> strings;
  auto status = options->cast(key, &strings, safety);
    if(status == pressio_options_key_set) {
    *size = strings.size();
    *values = static_cast<char**>(malloc(sizeof(char*)* (*size)));
    for (size_t i = 0; i < *size; ++i) {
      (*values)[i] = strndup(strings[i].c_str(), strings[i].size());
    }
  }
  return status;
}
enum pressio_options_key_status pressio_options_as_strings(struct pressio_options
      const* options, const char* key, size_t* size, char*** values) {
  return pressio_options_cast_strings(options, key, pressio_conversion_implicit, size, values);
}




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

char* pressio_options_to_string(struct pressio_options const* options) {
  if(options == nullptr) return nullptr;
  std::stringstream ss;
  ss << *options;
  auto const& str = ss.str();
  return strdup(str.c_str());
}


std::vector<compat::string_view> pressio_options::search(compat::string_view const& value) {
  std::vector<compat::string_view> order;
  //normalize the string
  auto size = value.size();
  const unsigned int has_leading_slash = !value.empty() && value.front() == '/';
  const unsigned int has_training_slash = !value.empty() && value.back() == '/';
  if(size >= 2) {
    if(has_leading_slash) --size;
    if(has_training_slash) --size;
  } else if(size == 1){
    if(has_leading_slash) --size;
  }
  const auto normalized = value.substr(has_leading_slash, size);

  
  //special case empty string
  if(normalized.empty()) {
    order.emplace_back("");
    return order;
  }

  order.reserve(std::count(std::begin(normalized), std::end(normalized), '/') + 2);
  bool done = false;
  auto len = std::string::npos;
  while(!done) {
    order.emplace_back(normalized.substr(0, len));

    len = normalized.rfind('/', len - 1);
    done = (len == std::string::npos);
  }
  order.emplace_back("");

  return order;
}
