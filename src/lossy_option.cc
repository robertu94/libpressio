#include <string>
#include <variant>
#include "lossy_option.h"
#include "lossy_options_impl.h"


extern "C" {
struct lossy_option* lossy_option_new() {
  return new lossy_option();
}
void lossy_option_free(struct lossy_option* options) {
  delete options;
}

#define lossy_option_define_type_impl(name, type) \
  struct lossy_option* lossy_option_new_##name(type value) { \
    return new lossy_option(value);\
  } \
  type lossy_option_get_##name(struct lossy_option const* option) { \
    return std::get<type>(option->option); \
  } \
  void lossy_option_set_##name(struct lossy_option* option, type value) { \
    option->option = value;\
  }

lossy_option_define_type_impl(uinteger, unsigned int)
lossy_option_define_type_impl(integer, int)
lossy_option_define_type_impl(float, float)
lossy_option_define_type_impl(double, double)
lossy_option_define_type_impl(userptr, void*)


//special case string
const char* lossy_option_get_string(struct lossy_option const* option) {
  return std::get<std::string>(option->option).c_str();
}

void lossy_option_set_string(struct lossy_option* option, const char* value) {
  option->option = value;
}

struct lossy_option* lossy_option_new_string(const char* value) {
  return new lossy_option(value);
}


lossy_option_type lossy_option::type() const {
  auto& o = option;
  if (std::holds_alternative<std::string>(o)) return lossy_option_charptr_type;
  else if (std::holds_alternative<int>(o)) return lossy_option_int32_type;
  else if (std::holds_alternative<unsigned int>(o)) return lossy_option_uint32_type;
  else if (std::holds_alternative<double>(o)) return lossy_option_double_type;
  else if (std::holds_alternative<float>(o)) return lossy_option_float_type;
  else if (std::holds_alternative<void*>(o)) return lossy_option_userptr_type;
  else return lossy_option_unset;
}

namespace {
  bool allow_explicit(lossy_conversion_safety safety) { return safety >= lossy_conversion_explicit; }
  bool allow_special(lossy_conversion_safety safety) { return safety >= lossy_conversion_explicit; }
}

std::optional<option_type> lossy_option::as(const enum lossy_option_type to_type, const enum lossy_conversion_safety safety) const {
  switch(type())
  {
    case lossy_option_double_type:
      {
      double d = std::get<double>(option);
      switch (to_type) {
        case lossy_option_double_type:
          return new lossy_option(d);
        case lossy_option_float_type:
            //narrowing
            if (allow_explicit(safety)) return option_type(static_cast<float>(d));
            else return std::nullopt;
        case lossy_option_int32_type:
            //narrowing
            if (allow_explicit(safety)) return option_type(static_cast<int>(d));
            else return std::nullopt;
        case lossy_option_uint32_type:
            //narrowing
            if (allow_explicit(safety)) return option_type(static_cast<unsigned int>(d));
            else return std::nullopt;
        case lossy_option_charptr_type:
            if (allow_special(safety)) return std::to_string(d);
            else return std::nullopt;
        default:
          return std::nullopt;
      }
      }
    case lossy_option_float_type:
      {
        float f = std::get<float>(option);
        switch(to_type) {
          case lossy_option_double_type:
              return option_type(static_cast<double>(f));
          case lossy_option_float_type:
              return option_type(f);
          case lossy_option_int32_type:
              //narrowing
              if (allow_explicit(safety)) return option_type(static_cast<int>(f));
              else return std::nullopt;
          case lossy_option_uint32_type:
              //narrowing
              if (allow_explicit(safety)) return option_type(static_cast<unsigned int>(f));
              else return std::nullopt;
        case lossy_option_charptr_type:
            if (allow_special(safety)) return std::to_string(f);
            else return std::nullopt;
          default:
            return std::nullopt;
        }
      }
    case lossy_option_int32_type:
      {
      int i = std::get<int>(option);
      switch(to_type) {
          case lossy_option_double_type:
            return option_type(static_cast<double>(i));
          case lossy_option_float_type:
            return option_type(static_cast<float>(i));
          case lossy_option_int32_type:
            return option_type(i);
          case lossy_option_uint32_type:
              //sign conversion
              if (allow_explicit(safety)) return new option_type(static_cast<unsigned int>(i));
              else return std::nullopt;
        case lossy_option_charptr_type:
            if (allow_special(safety)) return std::to_string(i);
            else return std::nullopt;
          default:
            return std::nullopt;
      }
      }
    case lossy_option_uint32_type:
      {
        unsigned int i = std::get<unsigned int>(option);
        switch(to_type) {
          case lossy_option_double_type:
            return option_type(static_cast<double>(i));
          case lossy_option_float_type:
            return option_type(static_cast<float>(i));
          case lossy_option_uint32_type:
              return option_type(i);
          case lossy_option_int32_type:
              //sign conversion
              if (allow_explicit(safety)) return option_type(static_cast<int>(i));
              else return std::nullopt;
        case lossy_option_charptr_type:
            if (allow_special(safety)) return std::to_string(i);
            else return std::nullopt;
          default:
            return std::nullopt;
        }
      }
    case lossy_option_charptr_type:
      {
        std::string const& s = std::get<std::string>(option);
        try {
        switch(to_type) {
          case lossy_option_double_type:
            if (allow_special(safety)) return option_type(std::stod(s));
            else return std::nullopt;
          case lossy_option_float_type:
            if (allow_special(safety)) return option_type(std::stod(s));
            else return std::nullopt;
          case lossy_option_int32_type:
            if (allow_special(safety)) return option_type(std::stoi(s));
            else return std::nullopt;
          case lossy_option_uint32_type:
            if (allow_special(safety)) return option_type(static_cast<unsigned int>(std::stoul(s)));
            else return std::nullopt;
          case lossy_option_charptr_type:
            return s;
          default:
            return std::nullopt;
        }
        } catch (std::invalid_argument const&) { return std::nullopt; }
        catch (std::out_of_range const&) {return std::nullopt;}
      }
    case lossy_option_userptr_type:
      if(to_type == lossy_option_userptr_type) return std::get<void*>(option);
      else return std::nullopt;
    default:
      return std::nullopt;
  }
}

enum lossy_option_type lossy_option_get_type(struct lossy_option const* option) {
  return option->type();
}

struct lossy_option* lossy_option_convert_implicit(struct lossy_option const* option, enum lossy_option_type type) {
  return lossy_option_convert(option, type, lossy_conversion_implicit);
}

struct lossy_option* lossy_option_convert(struct lossy_option const* option, enum lossy_option_type type, enum lossy_conversion_safety safety) {
  auto const& value = option->as(type, safety);
  if(value) return new lossy_option(value.value());
  else return nullptr;
}

}
