#include <string>
#include <sstream>
#include <stdexcept>
#include "libpressio_ext/cpp/dtype.h"
#include "pressio_option.h"
#include "libpressio_ext/cpp/userptr.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "std_compat/std_compat.h"


namespace {
  bool allow_explicit(pressio_conversion_safety safety) { return safety >= pressio_conversion_explicit; }
  bool allow_special(pressio_conversion_safety safety) { return safety >= pressio_conversion_special; }

  template <class From, class To, typename Enable = void>
  struct is_narrowing_conversion : public std::false_type{};

  template <class From, class To>
  struct is_narrowing_conversion<From, To, typename std::enable_if<sizeof(From) && sizeof(To)>::type> : public 
    std::integral_constant<bool, 
    (std::is_floating_point<From>::value && std::is_integral<To>::value) ||
    (std::is_floating_point<From>::value && std::is_floating_point<To>::value && sizeof(From) > sizeof(To)) ||
    (std::is_integral<From>::value && std::is_floating_point<To>::value) ||
    (std::is_integral<From>::value && std::is_integral<To>::value
           && (sizeof(From) > sizeof(To)
               || (std::is_signed<From>::value ? !std::is_signed<To>::value
                   : (std::is_signed<To>::value && sizeof(From) == sizeof(To))))) 
    >

  {
  };

  template <class To, class From>
  pressio_option convert_numeric(const From& option, const enum pressio_conversion_safety safety) {
    if(std::is_same<From,To>::value) {
      return pressio_option(option);
    } else if(is_narrowing_conversion<From,To>::value) {
      if (allow_explicit(safety)) {
        return pressio_option(static_cast<To>(option));
      } else {
        return {};
      }
    } else {
      return pressio_option(static_cast<To>(option));
    }
  }

  template <class From>
  pressio_option as_numeric(const pressio_option& option, const enum pressio_option_type to_type, const enum pressio_conversion_safety safety) {
      From d = option.get_value<From>();
      switch (to_type) {
        case pressio_option_bool_type:
          return convert_numeric<bool>(d, safety);
        case pressio_option_double_type:
          return convert_numeric<double>(d, safety);
        case pressio_option_float_type:
          return convert_numeric<float>(d, safety);
        case pressio_option_int8_type:
          return convert_numeric<int8_t>(d, safety);
        case pressio_option_uint8_type:
          return convert_numeric<uint8_t>(d, safety);
        case pressio_option_int16_type:
          return convert_numeric<int16_t>(d, safety);
        case pressio_option_uint16_type:
          return convert_numeric<uint16_t>(d, safety);
        case pressio_option_int32_type:
          return convert_numeric<int32_t>(d, safety);
        case pressio_option_uint32_type:
          return convert_numeric<uint32_t>(d, safety);
        case pressio_option_int64_type:
          return convert_numeric<int64_t>(d, safety);
        case pressio_option_uint64_type:
          return convert_numeric<uint64_t>(d, safety);
        case pressio_option_charptr_type:
            if (allow_special(safety)) return std::to_string(d);
            else return {};
        case pressio_option_charptr_array_type:
            if (allow_special(safety)) return std::vector<std::string>{std::to_string(d)};
            else return {};
        case pressio_option_data_type:
            if (allow_special(safety)) {
              auto ret = pressio_data::owning(libpressio::pressio_dtype_from_type<From>(), {1});
              *static_cast<From*>(ret.data()) = d;
              return ret;
            }
            else return {};
        default:
          return {};
    }
  }
}

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

pressio_option_define_type_impl(bool, bool)
pressio_option_define_type_impl(uinteger8, uint8_t)
pressio_option_define_type_impl(integer8, int8_t)
pressio_option_define_type_impl(uinteger16, uint16_t)
pressio_option_define_type_impl(integer16, int16_t)
pressio_option_define_type_impl(uinteger, uint32_t)
pressio_option_define_type_impl(integer, int32_t)
pressio_option_define_type_impl(uinteger64, uint64_t)
pressio_option_define_type_impl(integer64, int64_t)
pressio_option_define_type_impl(float, float)
pressio_option_define_type_impl(double, double)
pressio_option_define_type_impl(dtype, enum pressio_dtype)
pressio_option_define_type_impl(threadsafety, enum pressio_thread_safety)
pressio_option_define_type_impl(userptr, void*)

//special case userptr managed
struct pressio_option* pressio_option_new_userptr_managed(
    void* value,
    void* metadata,
    void(*deleter)(void*, void*),
    void(*copy)(void**, void**, const void*, const void*)) {
  return new pressio_option(userdata(value, metadata, deleter, copy));
}

//special case userptr managed
void pressio_option_set_userptr_managed(struct pressio_option* option,
    void* value,
    void* metadata,
    void(*deleter)(void*, void*),
    void(*copy)(void**, void**, const void*, const void*)) {
  option->set(userdata(value, metadata, deleter, copy));
}

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


struct pressio_option* pressio_option_new_strings(const char** values, size_t size) {
  std::vector<std::string> values_v(values, values+size);
  return new pressio_option(values_v);
}
const char** pressio_option_get_strings(struct pressio_option const* option, size_t* size) {
  auto const& value = option->get_value<std::vector<std::string>>();
  *size = value.size();
  auto ret = (const char**) malloc(sizeof(const char*)*value.size());
  for (size_t i = 0; i < *size; ++i) {
    ret[i] = strdup(value[i].c_str());
  }
  return ret;
}

void pressio_option_set_strings(struct pressio_option* option, const char** values, size_t size) {
  option->set(std::vector<std::string>(values, values+size));
}

enum pressio_options_key_status pressio_option_cast_set(struct pressio_option* lhs, struct pressio_option* rhs, enum pressio_conversion_safety safety) {
  return lhs->cast_set(*rhs, safety);
}

enum pressio_options_key_status pressio_option_as_set(struct pressio_option* lhs, struct pressio_option* rhs) {
  return lhs->cast_set(*rhs);
}


struct pressio_option* pressio_option_new_data(struct pressio_data* data) {
  return new pressio_option(*data);
}

pressio_data* pressio_option_get_data(struct pressio_option const* option) {
  return new pressio_data(option->get_value<pressio_data>());
}

void pressio_option_set_data(struct pressio_option* option, struct pressio_data* value) {
  return option->set(*value);
}

  void pressio_option::set(void* v) {
    option = compat::optional<userdata>(userdata(v));
  }

  bool pressio_option::operator==(pressio_option const& rhs) const {
    return option == rhs.option;
  }
  enum pressio_options_key_status pressio_option::cast_set(struct pressio_option const& rhs, enum pressio_conversion_safety safety) { 
    auto casted = rhs.as(type(), safety);
    if (casted.has_value()) {
      *this = std::move(casted);
      return pressio_options_key_set;
    } else {
      return pressio_options_key_exists; 
    }
  }
  void pressio_option::set_type(pressio_option_type type) {
    switch(type)
    {
      case pressio_option_charptr_type:
        option = compat::optional<std::string>();
        break;
      case pressio_option_userptr_type:
        option = compat::optional<userdata>();
        break;
      case pressio_option_bool_type:
        option = compat::optional<bool>();
        break;
      case pressio_option_int8_type:
        option = compat::optional<int8_t>();
        break;
      case pressio_option_uint8_type:
        option = compat::optional<uint8_t>();
        break;
      case pressio_option_int16_type:
        option = compat::optional<int16_t>();
        break;
      case pressio_option_uint16_type:
        option = compat::optional<uint16_t>();
        break;
      case pressio_option_int32_type:
        option = compat::optional<int32_t>();
        break;
      case pressio_option_uint32_type:
        option = compat::optional<uint32_t>();
        break;
      case pressio_option_int64_type:
        option = compat::optional<int64_t>();
        break;
      case pressio_option_uint64_type:
        option = compat::optional<uint64_t>();
        break;
      case pressio_option_float_type:
        option = compat::optional<float>();
        break;
      case pressio_option_double_type:
        option = compat::optional<double>();
        break;
      case pressio_option_unset_type:
        option = compat::monostate{};
        break;
      case pressio_option_charptr_array_type:
        option = compat::optional<std::vector<std::string>>();
        break;
      case pressio_option_data_type:
        option = compat::optional<pressio_data>();
        break;
      case pressio_option_threadsafety_type:
        option = compat::optional<pressio_thread_safety>();
        break;
      case pressio_option_dtype_type:
        option = compat::optional<pressio_dtype>();
        break;
    }
  }

bool pressio_option::has_value() const {
    if (holds_alternative<compat::monostate>()) return false;
    else {
      switch(type())
      {
        case pressio_option_bool_type:
          return (bool)get<bool>();
        case pressio_option_int8_type:
          return (bool)get<int8_t>();
        case pressio_option_uint8_type:
          return (bool)get<uint8_t>();
        case pressio_option_int16_type:
          return (bool)get<int16_t>();
        case pressio_option_uint16_type:
          return (bool)get<uint16_t>();
        case pressio_option_int32_type:
          return (bool)get<int32_t>();
        case pressio_option_uint32_type:
          return (bool)get<uint32_t>();
        case pressio_option_int64_type:
          return (bool)get<int64_t>();
        case pressio_option_uint64_type:
          return (bool)get<uint64_t>();
        case pressio_option_float_type:
          return (bool)get<float>();
        case pressio_option_double_type:
          return (bool)get<double>();
        case pressio_option_charptr_type:
          return (bool)get<std::string>();
        case pressio_option_userptr_type:
          return (bool)get<userdata>();
        case pressio_option_charptr_array_type:
          return (bool)get<std::vector<std::string>>();
        case pressio_option_data_type:
          return (bool)get<pressio_data>();
        case pressio_option_dtype_type:
          return (bool)get<pressio_dtype>();
        case pressio_option_threadsafety_type:
          return (bool)get<pressio_thread_safety>();
        case pressio_option_unset_type:
        default:
          return false;
      }
    }
  }

pressio_option_type pressio_option::type() const {
  if (holds_alternative<std::string>()) return pressio_option_charptr_type;
  else if (holds_alternative<int8_t>()) return pressio_option_int8_type;
  else if (holds_alternative<bool>()) return pressio_option_bool_type;
  else if (holds_alternative<uint8_t>()) return pressio_option_uint8_type;
  else if (holds_alternative<int16_t>()) return pressio_option_int16_type;
  else if (holds_alternative<uint16_t>()) return pressio_option_uint16_type;
  else if (holds_alternative<int32_t>()) return pressio_option_int32_type;
  else if (holds_alternative<uint32_t>()) return pressio_option_uint32_type;
  else if (holds_alternative<int64_t>()) return pressio_option_int64_type;
  else if (holds_alternative<uint64_t>()) return pressio_option_uint64_type;
  else if (holds_alternative<double>()) return pressio_option_double_type;
  else if (holds_alternative<float>()) return pressio_option_float_type;
  else if (holds_alternative<userdata>()) return pressio_option_userptr_type;
  else if (holds_alternative<std::vector<std::string>>()) return pressio_option_charptr_array_type;
  else if (holds_alternative<pressio_data>()) return pressio_option_data_type;
  else if (holds_alternative<pressio_dtype>()) return pressio_option_dtype_type;
  else if (holds_alternative<pressio_thread_safety>()) return pressio_option_threadsafety_type;
  else return pressio_option_unset_type;
}

pressio_option pressio_option::as(const enum pressio_option_type to_type, const enum pressio_conversion_safety safety) const {
  if(not has_value()) {
    return {};
  }
  switch(type())
  {
    case pressio_option_bool_type:
      return as_numeric<bool>(*this, to_type, safety);
    case pressio_option_double_type:
      return as_numeric<double>(*this, to_type, safety);
    case pressio_option_float_type:
      return as_numeric<float>(*this, to_type, safety);
    case pressio_option_int8_type:
      return as_numeric<int8_t>(*this, to_type, safety);
    case pressio_option_uint8_type:
      return as_numeric<uint8_t>(*this, to_type, safety);
    case pressio_option_int16_type:
      return as_numeric<int16_t>(*this, to_type, safety);
    case pressio_option_uint16_type:
      return as_numeric<uint16_t>(*this, to_type, safety);
    case pressio_option_int32_type:
      return as_numeric<int32_t>(*this, to_type, safety);
    case pressio_option_uint32_type:
      return as_numeric<uint32_t>(*this, to_type, safety);
    case pressio_option_int64_type:
      return as_numeric<int64_t>(*this, to_type, safety);
    case pressio_option_uint64_type:
      return as_numeric<uint64_t>(*this, to_type, safety);
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
          case pressio_option_int8_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(std::stoi(s));
            else return {};
          case pressio_option_bool_type:
            if (allow_special(safety)) return pressio_option(s=="true");
            else return {};
          case pressio_option_uint8_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint8_t>(std::stoul(s)));
            else return {};
          case pressio_option_int16_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<int16_t>(std::stoi(s)));
            else return {};
          case pressio_option_uint16_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint16_t>(std::stoul(s)));
            else return {};
          case pressio_option_int32_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<int32_t>(std::stol(s)));
            else return {};
          case pressio_option_uint32_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint32_t>(std::stoul(s)));
            else return {};
          case pressio_option_int64_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<int64_t>(std::stoll(s)));
            else return {};
          case pressio_option_uint64_type:
            if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint64_t>(std::stoull(s)));
            else return {};
          case pressio_option_threadsafety_type:
            if (allow_special(safety)) {
              if (s == "single") return pressio_option(pressio_thread_safety_single);
              else if (s == "multiple") return pressio_option(pressio_thread_safety_multiple);
              else if (s == "serialized") return pressio_option(pressio_thread_safety_serialized);
              else return {};
            }
            else return {};
          case pressio_option_dtype_type:
            if (allow_special(safety)) {
              if (s == "int8") return pressio_option(pressio_int8_dtype);
              else if (s == "uint8") return pressio_option(pressio_uint8_dtype);
              else if (s == "int16") return pressio_option(pressio_int16_dtype);
              else if (s == "uint16") return pressio_option(pressio_uint16_dtype);
              else if (s == "int32") return pressio_option(pressio_int32_dtype);
              else if (s == "uint32") return pressio_option(pressio_uint32_dtype);
              else if (s == "int64") return pressio_option(pressio_int64_dtype);
              else if (s == "uint64") return pressio_option(pressio_uint64_dtype);
              else if (s == "float") return pressio_option(pressio_float_dtype);
              else if (s == "double") return pressio_option(pressio_double_dtype);
              else if (s == "bool") return pressio_option(pressio_bool_dtype);
              else if (s == "byte") return pressio_option(pressio_byte_dtype);
              else return {};
            }
            else return {};
          case pressio_option_charptr_type:
            return s;
          case pressio_option_data_type:
              if (allow_special(safety)) return pressio_data{std::stod(s)}; 
              else return {};
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
            case pressio_option_int8_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(std::stoi(s));
              else return {};
            case pressio_option_bool_type:
              if (allow_special(safety)) return pressio_option(static_cast<bool>(s=="true"));
              else return {};
            case pressio_option_uint8_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint8_t>(std::stoul(s)));
              else return {};
            case pressio_option_int16_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<int16_t>(std::stoi(s)));
              else return {};
            case pressio_option_uint16_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint16_t>(std::stoul(s)));
              else return {};
            case pressio_option_int32_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<int32_t>(std::stol(s)));
              else return {};
            case pressio_option_uint32_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint32_t>(std::stoul(s)));
              else return {};
            case pressio_option_int64_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<int64_t>(std::stoll(s)));
              else return {};
            case pressio_option_uint64_type:
              if (allow_special(safety) && !s.empty()) return pressio_option(static_cast<uint64_t>(std::stoull(s)));
              else return {};
            case pressio_option_charptr_type:
              return s;
            case pressio_option_charptr_array_type:
                if (allow_special(safety)) return std::vector<std::string>{s};
                else return {};
            case pressio_option_data_type:
                if (allow_special(safety)) return pressio_data{std::stod(s)}; 
                else return {};
            default:
              return {};
          }
          } catch (std::invalid_argument const&) { return {}; }
          catch (std::out_of_range const&) {return {};}
        } else if (sa.size() > 1){
          try {
          switch(to_type){
            case pressio_option_data_type:
              {
                if(allow_special(safety)) {
                  std::vector<double> values(sa.size());
                  for (size_t i = 0; i < sa.size(); ++i) {
                    values[i] = stod(sa[i]);
                  }
                  return pressio_data::copy(pressio_double_dtype, values.data(), {sa.size()});
                }
              }
              break;
            default:
              return {};
          }
          } catch(std::invalid_argument&) {
            return {};
          } catch(std::out_of_range&) {
            return {};
          }
        }
        return {};
      }
    case pressio_option_threadsafety_type:
      if(to_type == pressio_option_threadsafety_type) return get_value<pressio_thread_safety>();
      if(to_type == pressio_option_charptr_type) {
        switch(get_value<pressio_thread_safety>()) {
          case pressio_thread_safety_serialized:
            return pressio_option(std::string("serialized"));
          case pressio_thread_safety_multiple:
            return pressio_option(std::string("multiple"));
          case pressio_thread_safety_single:
            return pressio_option(std::string("single"));
          default:
            return {};
        }
      }
      else return {};
    case pressio_option_dtype_type:
      if(to_type == pressio_option_dtype_type) return get_value<pressio_dtype>();
      if(to_type == pressio_option_charptr_type) {
        switch(get_value<pressio_dtype>()) {
          case pressio_bool_dtype:
            return pressio_option(std::string("bool"));
          case pressio_byte_dtype:
            return pressio_option(std::string("byte"));
          case pressio_float_dtype:
            return pressio_option(std::string("float"));
          case pressio_double_dtype:
            return pressio_option(std::string("double"));
          case pressio_int8_dtype:
            return pressio_option(std::string("int8"));
          case pressio_uint8_dtype:
            return pressio_option(std::string("uint8"));
          case pressio_int16_dtype:
            return pressio_option(std::string("int16"));
          case pressio_uint16_dtype:
            return pressio_option(std::string("uint16"));
          case pressio_int32_dtype:
            return pressio_option(std::string("int32"));
          case pressio_uint32_dtype:
            return pressio_option(std::string("uint32"));
          case pressio_int64_dtype:
            return pressio_option(std::string("int64"));
          case pressio_uint64_dtype:
            return pressio_option(std::string("uint64"));
          default:
            return {};
        }
      }
      else return {};
    case pressio_option_userptr_type:
      if(to_type == pressio_option_userptr_type) return get_value<void*>();
      else return {};
    //don't allow conversions from pressio_option_data_type for now
    case pressio_option_data_type:
      if(to_type == pressio_option_data_type) {
        return get_value<pressio_data>();
      } else return {};
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

char* pressio_option_to_string(struct pressio_option const* option) {
  std::stringstream ss;
  ss << *option;
  auto const& str = ss.str();
  return strdup(str.c_str());

}

}
