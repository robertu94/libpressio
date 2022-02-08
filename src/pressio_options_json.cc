#include <nlohmann/json.hpp>
#include <libpressio_ext/cpp/options.h>
#include <libpressio_ext/cpp/pressio.h>
#include <pressio_options.h>
#include <vector>
#include <stdexcept>
#include <sstream>

void to_json(nlohmann::json& j, pressio_data const& data){
  j["dims"] = data.dimensions();
  j["dtype"] = uint32_t{data.dtype()};
  j["values"] =  data.cast(pressio_double_dtype).to_vector<double>();
}

void to_json(nlohmann::json& j, pressio_option const& option){
  j["type"] = uint32_t{option.type()};
  if(not option.has_value()) {
    j["value"] = nullptr;
    return;
  }
  switch(option.type()) {
    case pressio_option_data_type:
      j["value"] = option.get_value<pressio_data>();
      break;
    case pressio_option_int8_type:
      j["value"] = option.get_value<int8_t>();
      break;
    case pressio_option_int16_type:
      j["value"] = option.get_value<int16_t>();
      break;
    case pressio_option_int32_type:
      j["value"] = option.get_value<int32_t>();
      break;
    case pressio_option_int64_type:
      j["value"] = option.get_value<int64_t>();
      break;
    case pressio_option_uint8_type:
      j["value"] = option.get_value<uint8_t>();
      break;
    case pressio_option_uint16_type:
      j["value"] = option.get_value<uint16_t>();
      break;
    case pressio_option_uint32_type:
      j["value"] = option.get_value<uint32_t>();
      break;
    case pressio_option_uint64_type:
      j["value"] = option.get_value<uint64_t>();
      break;
    case pressio_option_float_type:
      j["value"] = option.get_value<float>();
      break;
    case pressio_option_bool_type:
      j = option.get_value<bool>();
      break;
    case pressio_option_double_type:
      //javascript numbers are IEEE 754 double precision values
      //store them directly
      j = option.get_value<double>();
      break;
    case pressio_option_charptr_type:
      //strings can be represented directly in javascript
      //store them directly
      j = option.get_value<std::string>();
      break;
    case pressio_option_charptr_array_type:
      //lists of strings can be represented directly in javascript
      //store them directly
      j = option.get_value<std::vector<std::string>>();
      break;
    case pressio_option_unset_type:
      j["value"] = nullptr;
      break;
    case pressio_option_userptr_type:
      throw std::runtime_error("userptr and unset ptr types are not convertible to JSON");
  }
}
void to_json(nlohmann::json& j, pressio_options const& options){
  j = {};
  for(auto const& it: options) {
    auto const& key = std::get<0>(it);
    auto const& value = std::get<1>(it);
    if(value.type() != pressio_option_userptr_type) {
      j[key] = value;
    }
  }
}

void from_json(nlohmann::json const& j, pressio_data& data) {
  std::vector<double> values = j.at("values");
  std::vector<size_t> dims = j.at("dims");
  auto dtype = static_cast<pressio_dtype>(j.at("dtype"));

  data = pressio_data::nonowning(
      pressio_double_dtype,
      values.data(),
      dims).cast(dtype);
}

//returns the type of a nlohmann::json array by looking for the first entry and returning it's type
static nlohmann::json::value_t array_type(nlohmann::json const& j) {
  auto type = j.type();
  switch(type) {
    case nlohmann::json::value_t::array:
      if(j.size() > 0)
        return array_type(j.at(0));
      else
        return type;
    default:
      return type;
  }
}

template <class T>
static void flatten(nlohmann::json const& j, std::vector<T>& values, std::vector<size_t>& dims, size_t depth) {

  switch(j.type()) {
    case nlohmann::json::value_t::array:
      {
        size_t entries{0};
        if(dims.size() <= depth) {
          dims.emplace_back(0);
        }
        for (auto const& i : j) {
          flatten(i, values, dims, depth+1);
          ++entries;
        }
        dims[depth] = entries;
      }
      break;
    case nlohmann::json::value_t::number_float:
    case nlohmann::json::value_t::number_unsigned:
    case nlohmann::json::value_t::number_integer:
      values.emplace_back(j);
      break;
    default:
      throw std::runtime_error("other types are not supported");
      break;
  }
}


static void from_json_array(nlohmann::json const& j, pressio_option& option) {
  if(j.type() != nlohmann::json::value_t::array) {
    throw std::runtime_error("expected an array input");
  }
  const auto j_type = array_type(j);
  switch(j_type) {
    case nlohmann::json::value_t::string:
    case nlohmann::json::value_t::binary:
      option = j.get<std::vector<std::string>>();
      break;
    case nlohmann::json::value_t::number_integer:
    case nlohmann::json::value_t::number_unsigned:
    case nlohmann::json::value_t::number_float:
      {
        std::vector<double> values;
        std::vector<size_t> dims;
        flatten(j, values, dims, 0);
        option = pressio_data::copy(
              pressio_double_dtype,
              values.data(),
              dims
            );
      }
      break;
    case nlohmann::json::value_t::array:
      option.set_type(pressio_option_unset_type);
      break;
    default:
      throw std::runtime_error("unexpected array type");
      break;

  }

}

void from_json(nlohmann::json const& j, pressio_option& option) {
  switch(j.type()) {
    case nlohmann::json::value_t::binary:
    case nlohmann::json::value_t::string:
      option = j.get<std::string>();
      break;
    case nlohmann::json::value_t::object:
      {
        pressio_option_type dt = j.at("type").get<pressio_option_type>();
        if(j["value"].type() != nlohmann::json::value_t::null) {
          switch(dt) {
            case pressio_option_data_type:
              {
                pressio_data d;
                from_json(j.at("value"), d);
                option = d;
              }
              break;
            case pressio_option_int8_type:
              option = int8_t(j.at("value"));
              break;
            case pressio_option_int16_type:
              option = int16_t(j.at("value"));
              break;
            case pressio_option_int32_type:
              option = int32_t(j.at("value"));
              break;
            case pressio_option_int64_type:
              option = int64_t(j.at("value"));
              break;
            case pressio_option_uint8_type:
              option = uint8_t(j.at("value"));
              break;
            case pressio_option_bool_type:
              option = bool(j.at("value"));
              break;
            case pressio_option_uint16_type:
              option = uint16_t(j.at("value"));
              break;
            case pressio_option_uint32_type:
              option = uint32_t(j.at("value"));
              break;
            case pressio_option_uint64_type:
              option = uint64_t(j.at("value"));
              break;
            case pressio_option_float_type:
              option = float(j.at("value"));
              break;
            case pressio_option_double_type:
              option = double(j.at("value"));
              break;
            case pressio_option_charptr_type:
              option = j.at("value").get<std::string>();
              break;
            case pressio_option_charptr_array_type:
              option = std::vector<std::string>{j.at("value")};
              break;
            case pressio_option_unset_type:
              option = {};
              break;
            case pressio_option_userptr_type:
              throw std::runtime_error("userptr and unset ptr types are not convertible to nlohmann::json");
          }
        } else {
          switch(dt) {
            case pressio_option_data_type:
            case pressio_option_uint8_type:
            case pressio_option_bool_type:
            case pressio_option_uint16_type:
            case pressio_option_uint32_type:
            case pressio_option_uint64_type:
            case pressio_option_int8_type:
            case pressio_option_int16_type:
            case pressio_option_int32_type:
            case pressio_option_int64_type:
            case pressio_option_float_type:
            case pressio_option_double_type:
            case pressio_option_charptr_array_type:
            case pressio_option_charptr_type:
            case pressio_option_userptr_type:
            case pressio_option_unset_type:
              option.set_type(dt);
              break;
            default:
              throw std::runtime_error("unknown types are not convertible to nlohmann::json");
          }

        }
      }
      break;
    case nlohmann::json::value_t::number_float:
    case nlohmann::json::value_t::number_integer:
    case nlohmann::json::value_t::number_unsigned:
      option = j.get<double>();
      break;
    case nlohmann::json::value_t::array:
      from_json_array(j, option);
      break;
    case nlohmann::json::value_t::null:
      option = {};
      break;
    case nlohmann::json::value_t::boolean:
      option = j.get<bool>();
      break;
    case nlohmann::json::value_t::discarded:
      break;
      
  }
}


void from_json(nlohmann::json const& j, pressio_options& options) {
  pressio_option o;
  for (auto const& i: j.items()) {
    from_json(i.value(), o);
    options.set(i.key(), o);
  }
}



extern "C" {
  struct pressio_options* pressio_options_new_json(struct pressio* library, const char* json) {
    pressio_options* options = nullptr;
    try {
      nlohmann::json parsed = nlohmann::json::parse(json);
      options = new pressio_options(parsed.get<pressio_options>());
    } catch (nlohmann::detail::out_of_range& ex) {
      if(library) {
        library->set_error(2, ex.what());
      }
    } catch (nlohmann::detail::parse_error& ex) {
      if(library) {
        std::stringstream err_msg;
        err_msg << ex.what() << "\n" << json;
        library->set_error(2, err_msg.str());
      }
    } catch (std::runtime_error& ex) {
      if(library) {
        library->set_error(1, ex.what());
      }
    }
    return options;
  }

  char* pressio_options_to_json(struct pressio* library, struct pressio_options const* options) {
    char* ret = nullptr;
    try {
      nlohmann::json jstr = *options;
      std::string str = jstr.dump();
      ret = strdup(str.c_str());
    } catch (std::runtime_error& ex) {
      if(library) {
        library->set_error(1, ex.what());
      }
    }
    return ret;
  }


}

