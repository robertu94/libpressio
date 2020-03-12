#include <string>
#include "pressio_option.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/compat/std_compat.h"


extern "C" {
struct pressio_option* pressio_option_new() {
  return new pressio_option();
}
void pressio_option_free(struct pressio_option* options) {
  delete options;
}

bool pressio_option_has_value(struct pressio_option const* option) {
  return option->has_value();
}

#define pressio_option_define_type_impl(name, type) \
  struct pressio_option* pressio_option_new_##name(type value) { \
    return new pressio_option(value);\
  } \
  type pressio_option_get_##name(struct pressio_option const* option) { \
    return option->get_value<type>(); \
  } \
  void pressio_option_set_##name(struct pressio_option* option, type value) { \
    option->set(value);\
  }

pressio_option_define_type_impl(uinteger, unsigned int)
pressio_option_define_type_impl(integer, int)
pressio_option_define_type_impl(float, float)
pressio_option_define_type_impl(double, double)
pressio_option_define_type_impl(userptr, void*)


//special case string
const char* pressio_option_get_string(struct pressio_option const* option) {
  auto const& value = option->get_value<std::string>();
  return value.c_str();
}

void pressio_option_set_string(struct pressio_option* option, const char* value) {
  option->set(std::string(value));
}

struct pressio_option* pressio_option_new_string(const char* value) {
  return new pressio_option(std::string(value));
}

enum pressio_options_key_status pressio_option_cast_set(struct pressio_option* lhs, struct pressio_option* rhs, enum pressio_conversion_safety safety) {
  return lhs->cast_set(*rhs, safety);
}

enum pressio_options_key_status pressio_option_as_set(struct pressio_option* lhs, struct pressio_option* rhs) {
  return lhs->cast_set(*rhs);
}



pressio_option_type pressio_option::type() const {
  if (holds_alternative<std::string>()) return pressio_option_charptr_type;
  else if (holds_alternative<int>()) return pressio_option_int32_type;
  else if (holds_alternative<unsigned int>()) return pressio_option_uint32_type;
  else if (holds_alternative<double>()) return pressio_option_double_type;
  else if (holds_alternative<float>()) return pressio_option_float_type;
  else if (holds_alternative<void*>()) return pressio_option_userptr_type;
  else if (holds_alternative<std::vector<std::string>>()) return pressio_option_charptr_array_type;
  else if (holds_alternative<pressio_data>()) return pressio_option_data_type;
  else return pressio_option_unset_type;
}

namespace {
  bool allow_explicit(pressio_conversion_safety safety) { return safety >= pressio_conversion_explicit; }
  bool allow_special(pressio_conversion_safety safety) { return safety >= pressio_conversion_explicit; }
}

pressio_option pressio_option::as(const enum pressio_option_type to_type, const enum pressio_conversion_safety safety) const {
  switch(type())
  {
    case pressio_option_double_type:
      {
      double d = get_value<double>();
      switch (to_type) {
        case pressio_option_double_type:
          return pressio_option(d);
        case pressio_option_float_type:
            //narrowing
            if (allow_explicit(safety)) return pressio_option(static_cast<float>(d));
            else return {};
        case pressio_option_int32_type:
            //narrowing
            if (allow_explicit(safety)) return pressio_option(static_cast<int>(d));
            else return {};
        case pressio_option_uint32_type:
            //narrowing
            if (allow_explicit(safety)) return pressio_option(static_cast<unsigned int>(d));
            else return {};
        case pressio_option_charptr_type:
            if (allow_special(safety)) return std::to_string(d);
            else return {};
        case pressio_option_charptr_array_type:
            if (allow_special(safety)) return std::vector<std::string>{std::to_string(d)};
            else return {};
        case pressio_option_data_type:
            if (allow_special(safety)) {
              auto ret = pressio_data::owning(pressio_double_dtype, {1});
              *static_cast<double*>(ret.data()) = d;
              return ret;
            }
            else return {};
        default:
          return {};
      }
      }
    case pressio_option_float_type:
      {
        float f = get_value<float>();
        switch(to_type) {
          case pressio_option_double_type:
              return pressio_option(static_cast<double>(f));
          case pressio_option_float_type:
              return pressio_option(f);
          case pressio_option_int32_type:
              //narrowing
              if (allow_explicit(safety)) return pressio_option(static_cast<int>(f));
              else return {};
          case pressio_option_uint32_type:
              //narrowing
              if (allow_explicit(safety)) return pressio_option(static_cast<unsigned int>(f));
              else return {};
        case pressio_option_charptr_type:
            if (allow_special(safety)) return pressio_option(std::to_string(f));
            else return {};
        case pressio_option_charptr_array_type:
            if (allow_special(safety)) return std::vector<std::string>{std::to_string(f)};
            else return {};
        case pressio_option_data_type:
            if (allow_special(safety)) {
              auto ret = pressio_data::owning(pressio_float_dtype, {1});
              *static_cast<float*>(ret.data()) = f;
              return ret;
            }
            else return {};
          default:
            return {};
        }
      }
    case pressio_option_int32_type:
      {
      int i = get_value<int>();
      switch(to_type) {
          case pressio_option_double_type:
            return pressio_option(static_cast<double>(i));
          case pressio_option_float_type:
            return pressio_option(static_cast<float>(i));
          case pressio_option_int32_type:
            return pressio_option(i);
          case pressio_option_uint32_type:
              //sign conversion
              if (allow_explicit(safety)) return pressio_option(static_cast<unsigned int>(i));
              else return {};
        case pressio_option_charptr_type:
            if (allow_special(safety)) return std::to_string(i);
            else return {};
        case pressio_option_charptr_array_type:
            if (allow_special(safety)) return std::vector<std::string>{std::to_string(i)};
            else return {};
        case pressio_option_data_type:
            if (allow_special(safety)) {
              auto ret = pressio_data::owning(pressio_int32_dtype, {1});
              *static_cast<int32_t*>(ret.data()) = i;
              return ret;
            } else {
              return {};
            }
          default:
            return {};
      }
      }
    case pressio_option_uint32_type:
      {
        unsigned int i = get_value<unsigned int>();
        switch(to_type) {
          case pressio_option_double_type:
            return pressio_option(static_cast<double>(i));
          case pressio_option_float_type:
            return pressio_option(static_cast<float>(i));
          case pressio_option_uint32_type:
              return pressio_option(i);
          case pressio_option_int32_type:
              //sign conversion
              if (allow_explicit(safety)) return pressio_option(static_cast<int>(i));
              else return {};
        case pressio_option_charptr_type:
            if (allow_special(safety)) return std::to_string(i);
            else return {};
        case pressio_option_charptr_array_type:
            if (allow_special(safety)) return std::vector<std::string>{std::to_string(i)};
            else return {};
        case pressio_option_data_type:
            if (allow_special(safety)) {
              auto ret = pressio_data::owning(pressio_uint32_dtype, {1});
              *static_cast<uint32_t*>(ret.data()) = i;
              return ret;
            }else {
              return {};
            }

          default:
            return {};
        }
      }
    case pressio_option_charptr_type:
      {
        std::string const& s = get_value<std::string>();
        try {
        switch(to_type) {
          case pressio_option_double_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(std::stod(s));
            else return {};
          case pressio_option_float_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(std::stof(s));
            else return {};
          case pressio_option_int32_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(std::stoi(s));
            else return {};
          case pressio_option_uint32_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<unsigned int>(std::stoul(s)));
            else return {};
          case pressio_option_charptr_type:
            return s;
          case pressio_option_charptr_array_type:
              if (allow_special(safety)) return std::vector<std::string>{s};
              else return {};
          default:
            return {};
        }
        } catch (std::invalid_argument const&) { return {}; }
        catch (std::out_of_range const&) {return {};}
      }
    case pressio_option_charptr_array_type:
      {
        std::vector<std::string> const& sa = get_value<std::vector<std::string>>();
        if(to_type == pressio_option_charptr_array_type) return pressio_option(sa);
        if(sa.size() == 1) {
          std::string const& s = sa.front();
          try {
          switch(to_type) {
            case pressio_option_double_type:
              if (allow_special(safety)) return pressio_option(std::stod(s));
              else return {};
            case pressio_option_float_type:
              if (allow_special(safety)) return pressio_option(std::stof(s));
              else return {};
            case pressio_option_int32_type:
              if (allow_special(safety)) return pressio_option(std::stoi(s));
              else return {};
            case pressio_option_uint32_type:
              if (allow_special(safety)) return pressio_option(static_cast<unsigned int>(std::stoul(s)));
              else return {};
            case pressio_option_charptr_type:
              return s;
            case pressio_option_charptr_array_type:
                if (allow_special(safety)) return std::vector<std::string>{s};
                else return {};
            default:
              return {};
          }
          } catch (std::invalid_argument const&) { return {}; }
          catch (std::out_of_range const&) {return {};}
        } else {
          return {};
        }
      }
    case pressio_option_userptr_type:
      if(to_type == pressio_option_userptr_type) return get_value<void*>();
      else return {};
    //don't allow conversions from pressio_option_data_type for now
    default:
      return {};
  }
}

enum pressio_option_type pressio_option_get_type(struct pressio_option const* option) {
  return option->type();
}

void pressio_option_set_type(struct pressio_option* option, enum pressio_option_type type) {
  return option->set_type(type);
}

struct pressio_option* pressio_option_convert_implicit(struct pressio_option const* option, enum pressio_option_type type) {
  return pressio_option_convert(option, type, pressio_conversion_implicit);
}

struct pressio_option* pressio_option_convert(struct pressio_option const* option, enum pressio_option_type type, enum pressio_conversion_safety safety) {
  auto const& value = option->as(type, safety);
  if(value.has_value()) return new pressio_option(value);
  else return nullptr;
}

}
