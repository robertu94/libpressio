#include <string>
#include <variant>
#include "lossy_option.h"
#include "lossy_options_impl.h"


extern "C" {
void lossy_option_free(struct lossy_option* options) {
  delete options;
}

unsigned int lossy_option_get_uinteger(struct lossy_option const* option) {
  return std::get<unsigned int>(option->option);
}
int lossy_option_get_integer(struct lossy_option const* option) {
  return std::get<int>(option->option);
}
double lossy_option_get_double(struct lossy_option const* option) {
  return std::get<double>(option->option);
}
float lossy_option_get_float(struct lossy_option const* option) {
  return std::get<float>(option->option);
}
const char* lossy_option_get_string(struct lossy_option const* option) {
  return std::get<std::string>(option->option).c_str();
}
void* lossy_option_get_userptr(struct lossy_option const* option) {
  return std::get<void*>(option->option);
}
void lossy_option_set_integer(struct lossy_option* option, int value) {
  option->option = value;
}
void lossy_option_set_double(struct lossy_option* option, double value) {
  option->option = value;
}
void lossy_option_set_float(struct lossy_option* option, float value) {
  option->option = value;
}
void lossy_option_set_string(struct lossy_option* option, const char* value) {
  option->option = value;
}
void lossy_option_set_userptr(struct lossy_option* option, void* value) {
  option->option = value;
}

lossy_option_type lossy_option_get_type(struct lossy_option const* option) {
  auto& o = option->option;
  if (std::holds_alternative<std::string>(o)) return lossy_option_charptr_type;
  else if (std::holds_alternative<int>(o)) return lossy_option_int32_type;
  else if (std::holds_alternative<unsigned int>(o)) return lossy_option_uint32_type;
  else if (std::holds_alternative<double>(o)) return lossy_option_double_type;
  else if (std::holds_alternative<float>(o)) return lossy_option_float_type;
  else if (std::holds_alternative<void*>(o)) return lossy_option_userptr_type;
  else return lossy_option_unset;

}

}
