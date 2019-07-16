#include <variant>
#include <map>
#include <string>
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


void lossy_options_set_userptr(struct lossy_options* options, const char* key, void* value) {
  options->set(key, value);
}
void lossy_options_set_uinteger(struct lossy_options* options, const char* key, unsigned int value) {
  options->set(key, value);
}
void lossy_options_set_integer(struct lossy_options* options, const char* key, int value) {
  options->set(key, value);
}
void lossy_options_set_string(struct lossy_options* options, const char* key, const char* value) {
  options->set(key, value);
}
void lossy_options_set_float(struct lossy_options* options, const char* key, float value) {
  options->set(key, value);
}
void lossy_options_set_double(struct lossy_options* options, const char* key, double value) {
  options->set(key, value);
}
void lossy_options_clear(struct lossy_options* options, const char* key) {
  options->set(key, std::monostate());
}
enum lossy_options_key_status lossy_options_get_userptr(struct lossy_options const* options, const char* key, void** value) {
  return options->get(key, value);
}

enum lossy_options_key_status lossy_options_get_uinteger(struct lossy_options const* options, const char* key, unsigned int* value) {
  return options->get(key, value);
}
enum lossy_options_key_status lossy_options_get_integer(struct lossy_options const* options, const char* key, int* value) {
  return options->get(key, value);
}
enum lossy_options_key_status lossy_options_get_string(struct lossy_options const* options, const char* key, const char** value) {
  return options->get(key, value);
  }
enum lossy_options_key_status lossy_options_get_double(struct lossy_options const* options, const char* key, double* value) {
  return options->get(key, value);
}
enum lossy_options_key_status lossy_options_get_float(struct lossy_options const* options, const char* key, float* value) {
  return options->get(key, value);
}

lossy_options_key_status lossy_options_exists(struct lossy_options const* options, const char* key) {
  return options->key_status(key);
}

lossy_option* lossy_options_get(struct lossy_options const* options, const char* key) {
  return new lossy_option(options->get(key));
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
