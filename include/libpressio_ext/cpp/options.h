#ifndef PRESSIO_OPTIONS_CPP
#define PRESSIO_OPTIONS_CPP

#include <optional>
#include <variant>
#include <string>
#include <map>
#include <type_traits>
#include "pressio_options.h"
#include "pressio_option.h"

/**
 * \file
 * \brief C++ pressio_options and pressio_option interfaces
 */

namespace {

  /** defines constants to convert between types and pressio_option_*_type */
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

using option_type = std::variant<std::monostate,
      std::optional<int>,
      std::optional<unsigned int>,
      std::optional<float>,
      std::optional<double>,
      std::optional<std::string>,
      std::optional<void*>
      >;
}

/**
 * represents a dynamically typed object
 */
struct pressio_option final {
  /** constructs an option without type or value */
  pressio_option()=default;

  /** constructs an option that holds the specified value
   * \param[in] value the value the option is to hold
   * */
  template<class T>
  pressio_option(T value): option(std::optional<T>(value)) {}


  /** returns a pressio option that has the appropriate type if the conversion is allowed for the specified safety
   * \param[in] type the type to convert to
   * \param[in] safety the level of conversion safety to use \see pressio_conversion_safety
   * \returns a empty option if no conversion is possible or the converted value as an option
   */
  pressio_option as(const enum pressio_option_type type, const enum pressio_conversion_safety safety = pressio_conversion_implicit) const;

  /** 
   * \returns the type currently held by the option
   */
  enum pressio_option_type type() const;

  /** 
   * \returns returns true if the option holds the current type
   */
  template <class T, std::enable_if_t<!std::is_same_v<T,std::monostate>,int> = 0>
  bool holds_alternative() const {
    return std::holds_alternative<std::optional<T>>(option);
  }

  /** Specialization for the std::monostate singleton
   * \returns true if the option has no specified type or value
   */
  template <class T, std::enable_if_t<std::is_same_v<T,std::monostate>,int> = 0>
  bool holds_alternative() const {
    return std::holds_alternative<std::monostate>(option);
  }

  /** 
   * \returns a std::optional which holds a value if the option has one or an empty optional otherwise
   */
  template <class T>
  std::optional<T> const& get() const{
    return std::get<std::optional<T>>(option);
  }

  /** 
   * This function has unspecified effects if the option contains no value \see has_value
   * \returns the value contained by the option
   */
  template <class T>
  T const& get_value() const{
    return get<T>().value();
  }

  /**
   * \returns true if the option holds a value or false otherwise
   */
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
        default:
          return false;
      }
    }
  }

  /**
   * set the option to a new value
   * \param[in] v the value to set the option to
   */
  template <class T>
  void set(T v) {
    option = std::optional(v);
  }

  /**
   * set only the type of the option
   * \param[in] type the type to set the option to
   */
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

  /**
   * converts rhs according the conversion safety specified to the type of the option and stores the result if the cast succeeds
   * \param[in] rhs the option to assign to this option
   * \param[in] safety the specified safety to use \see pressio_conversion_safety
   */
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

/** specialization for option to reset the type to hold no type or value
 * \param[in] value the monostate singleton
 * */
template<>
pressio_option::pressio_option(std::monostate value);

/**
 * represents a map of dynamically typed objects
 */
struct pressio_options final {

  /**
   * checks the status of a key in a option set
   * \returns pressio_options_key_does_not_exist if the key does not exist
   *          pressio_options_key_exists if the key exists but has no value
   *          pressio_options_key_set if the key exists and is set
   */
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

  /**
   * sets a key to the specified value
   * \param[in] key the key to use
   * \param[in] value the value to use
   */
  void set(std::string const& key,  pressio_option const& value) {
    options[key] = value;
  }

  /**
   * converts value according the conversion safety to the type of the option at the stored key specified and stores the result if the cast succeeds
   * \param[in] key the option key
   * \param[in] value the option to assign to this option
   * \param[in] safety the specified safety to use \see pressio_conversion_safety
   */
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

  /**
   * set only the type of the option
   * \param[in] key which option to set
   * \param[in] type the type to set the option to
   */
  void set_type(std::string const& key, pressio_option_type type) {
    options[key].set_type(type);
  }

  /**
   * \param[in] key which option to get
   * \returns the option at the specified key
   */
  pressio_option const& get(std::string const& key) const {
    return options.at(key);
  }

  /**
   * gets a key if it is set and stores it into the pointer value
   * \param[in] key the option to retrieve
   * \param[out] value the value that is in the option
   * \returns pressio_options_key_does_not_exist if the key does not exist
   *          pressio_options_key_exists if the key exists but has no value
   *          pressio_options_key_set if the key exists and is set
   */
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

  /**
   * gets a key if it is set, attepts to cast it the specified type and stores it into the pointer value
   * \param[in] key the option to retrieve
   * \param[out] value the value that is in the option
   * \param[in] safety the level of conversions to allow \see pressio_conversion_safety
   * \returns pressio_options_key_does_not_exist if the key does not exist
   *          pressio_options_key_exists if the key exists but has no value
   *          pressio_options_key_set if the key exists and is set
   */
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

  /**
   * \returns an begin iterator over the keys
   */
  auto begin() const {
    return std::begin(options);
  }
  /**
   * \returns an end iterator over the keys
   */
  auto end() const {
    return std::end(options);
  }

  /**
   * \returns an begin iterator over the keys
   */
  auto begin() {
    return std::begin(options);
  }
  /**
   * \returns an end iterator over the keys
   */
  auto end() {
    return std::end(options);
  }

  /**
   * type of the returned iterator
   */
  using iterator = std::map<std::string, pressio_option>::iterator;

  /**
   * type of the values in the map
   */
  using value_type = std::map<std::string, pressio_option>::value_type;

  /**
   * function to insert new values into the map
   */
  iterator insert(iterator it, value_type const& value) {
    return options.insert(it, value);
  }


  private:
  std::map<std::string, pressio_option> options;
};

//special case for strings
/**
 * special case for strings for memory management reasons \see pressio_options::get
 */
template <>
enum pressio_options_key_status pressio_options::get(std::string const& key, const char** value) const;

/**
 * special case for strings for memory management reasons \see pressio_options::cast
 */
template <>
enum pressio_options_key_status pressio_options::cast(std::string const& key, char** value, enum pressio_conversion_safety safety) const;

#endif
