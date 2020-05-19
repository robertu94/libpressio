#ifndef PRESSIO_OPTIONS_CPP
#define PRESSIO_OPTIONS_CPP

#include <string>
#include <map>
#include <type_traits>
#include <vector>
#include <initializer_list>
#include "pressio_options.h"
#include "pressio_option.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/compat/std_compat.h"

/**
 * \file
 * \brief C++ pressio_options and pressio_option interfaces
 */

namespace {
using option_type = compat::variant<compat::monostate,
      compat::optional<int>,
      compat::optional<unsigned int>,
      compat::optional<float>,
      compat::optional<double>,
      compat::optional<std::string>,
      compat::optional<void*>,
      compat::optional<std::vector<std::string>>,
      compat::optional<pressio_data>
      >;


}

/** defines constants to convert between types and pressio_option_*_type */
template <class T>
constexpr enum pressio_option_type pressio_type_to_enum() {
  return std::is_same<T, int>() ? pressio_option_int32_type :
    std::is_same<T,unsigned int>() ? pressio_option_uint32_type :
    std::is_same<T,float>() ? pressio_option_float_type :
    std::is_same<T,double>() ? pressio_option_double_type :
    std::is_same<T,std::string>() ? pressio_option_charptr_type :
    std::is_same<T,const char*>() ? pressio_option_charptr_type :
    std::is_same<T,const char**>() ? pressio_option_charptr_array_type :
    std::is_same<T,std::vector<std::string>>() ? pressio_option_charptr_array_type :
    std::is_same<T,pressio_data>() ? pressio_option_data_type :
    pressio_option_userptr_type;
    ;
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
  template<class T, typename = typename std::enable_if<
    !std::is_same<T, pressio_conversion_safety>::value &&
    !std::is_same<T, pressio_option>::value &&
    !std::is_same<T, const char*>::value &&
    !std::is_same<T, compat::monostate>::value
    >::type>
  pressio_option(compat::optional<T> const& value): option(value) {}

  /** constructs an option that holds the specified value
   * \param[in] value the value the option is to hold
   * */
  template<class T, typename = typename std::enable_if<
    !std::is_same<T, pressio_conversion_safety>::value &&
    !std::is_same<T, pressio_option>::value &&
    !std::is_same<T, const char*>::value &&
    !std::is_same<T, compat::monostate>::value
    >::type>
  pressio_option(T const& value): option(compat::optional<T>(value)) {}

  /** specialization for option to reset the type to hold no type or value
   * \param[in] value the monostate singleton
   * */
  pressio_option(compat::monostate value): option(value) { }

  /** specialization for strings to be compatable with c++11
   * \param[in] value the monostate singleton
   * */
  pressio_option(const char* value): option(std::string(value)) { }

  /** 
   * create an option from a std::initializer_list
   */

  template<class T>
  pressio_option(std::initializer_list<T> ul): option(ul) {}

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
  template <class T, typename std::enable_if<!std::is_same<T,compat::monostate>::value,int>::type = 0>
  bool holds_alternative() const {
    return compat::holds_alternative<compat::optional<T>>(option);
  }

  /** Specialization for the compat::monostate singleton
   * \returns true if the option has no specified type or value
   */
  template <class T, typename std::enable_if<std::is_same<T,compat::monostate>::value,int>::type = 0>
  bool holds_alternative() const {
    return compat::holds_alternative<compat::monostate>(option);
  }

  /** 
   * \returns a std::optional which holds a value if the option has one or an empty optional otherwise
   */
  template <class T>
  compat::optional<T> const& get() const{
    return compat::get<compat::optional<T>>(option);
  }

  /** 
   * This function has unspecified effects if the option contains no value \see has_value
   * \returns the value contained by the option
   */
  template <class T>
  T const& get_value() const{
    return *get<T>();
  }

  /**
   * \returns true if the option holds a value or false otherwise
   */
  bool has_value() const {
    if (holds_alternative<compat::monostate>()) return false;
    else {
      switch(type())
      {
        case pressio_option_int32_type:
          return (bool)get<int>();
        case pressio_option_uint32_type:
          return (bool)get<unsigned int>();
        case pressio_option_float_type:
          return (bool)get<float>();
        case pressio_option_double_type:
          return (bool)get<double>();
        case pressio_option_charptr_type:
          return (bool)get<std::string>();
        case pressio_option_userptr_type:
          return (bool)get<void*>();
        case pressio_option_charptr_array_type:
          return (bool)get<std::vector<std::string>>();
        case pressio_option_data_type:
          return (bool)get<pressio_data>();
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
    option = compat::optional<T>(v);
  }

  /**
   * set only the type of the option
   * \param[in] type the type to set the option to
   */
  void set_type(pressio_option_type type) {
    switch(type)
    {
      case pressio_option_charptr_type:
        option = compat::optional<std::string>();
        break;
      case pressio_option_userptr_type:
        option = compat::optional<void*>();
        break;
      case pressio_option_int32_type:
        option = compat::optional<int>();
        break;
      case pressio_option_uint32_type:
        option = compat::optional<unsigned int>();
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
      *this = std::move(casted);
      return pressio_options_key_set;
    } else {
      return pressio_options_key_exists; 
    }
  }

  private:
  option_type option;
};


/**
 * represents a map of dynamically typed objects
 */
struct pressio_options final {

  /** type of the keys for pressio_options, useful for lua */
  using key_type = std::string;
  /** type of the mapped_type for pressio_options, useful for lua */
  using mapped_type = pressio_option;

  /** create an empty pressio_options structure */
  pressio_options()=default;
  /** copy a pressio_options structure
   *
   * \param[in] rhs the structure to copy from
   * */
  pressio_options(pressio_options const& rhs)=default;
  /** move a pressio_options structure
   *
   * \param[in] rhs the structure to move from
   * */
  pressio_options(pressio_options && rhs) noexcept=default;
  /** copy a pressio_options structure
   *
   * \param[in] rhs the structure to copy from
   * */
  pressio_options& operator=(pressio_options const& rhs)=default;
  /** move a pressio_options structure 
   *
   * \param[in] rhs the structure to move from
   * */
  pressio_options& operator=(pressio_options && rhs) noexcept=default;

  /**
   * create a literal pressio_options structure from a std::initializer_list
   *
   * \param[in] opts the options to put into the map
   */
  pressio_options(std::initializer_list<std::pair<const std::string, pressio_option>> opts): options(opts) {}

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
   * checks the status of a key in a option set with a given name
   * \returns pressio_options_key_does_not_exist if the key does not exist
   *          pressio_options_key_exists if the key exists but has no value
   *          pressio_options_key_set if the key exists and is set
   */
  pressio_options_key_status key_status(std::string const& name, std::string const& key) const {
    return key_status(format_name(name, key));
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
   * sets a key to the specified value
   * \param[in] name the name to use
   * \param[in] key the key to use
   * \param[in] value the value to use
   */
  void set(std::string const& name, std::string const& key,  pressio_option const& value) {
    set(format_name(name, key), value);
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
   * converts value according the conversion safety to the type of the option at the stored key specified and stores the result if the cast succeeds
   * \param[in] name the name to use
   * \param[in] key the option key
   * \param[in] value the option to assign to this option
   * \param[in] safety the specified safety to use \see pressio_conversion_safety
   */
  enum pressio_options_key_status cast_set(std::string const& name, std::string const& key,  pressio_option const& value, enum pressio_conversion_safety safety= pressio_conversion_implicit) {
    return cast_set(format_name(name, key), value, safety);
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
   * set only the type of the option
   * \param[in] name the name to use
   * \param[in] key which option to set
   * \param[in] type the type to set the option to
   */
  void set_type(std::string const& name, std::string const& key, pressio_option_type type) {
    set_type(format_name(name, key), type);
  }

  /**
   * \param[in] key which option to get
   * \returns the option at the specified key
   */
  pressio_option const& get(std::string const& key) const {
    return options.at(key);
  }

  /**
   * \param[in] name the name to use
   * \param[in] key which option to get
   * \returns the option at the specified key
   */
  pressio_option const& get(std::string const& name, std::string const& key) const {
    auto prefix_key = format_name(name, key);
    if(options.find(prefix_key) != options.end()) return get(prefix_key);
    return get(key);
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
  enum pressio_options_key_status get(std::string const& key, compat::optional<PointerType>* value) const {
    using ValueType = typename std::remove_pointer<PointerType>::type;
    switch(key_status(key)){
      case pressio_options_key_set:
        {
          auto variant = get(key);
          if (variant.holds_alternative<ValueType>()) { 
            *value = variant.get<ValueType>();
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
   * gets a key if it is set and stores it into the pointer value
   * \param[in] key the option to retrieve
   * \param[out] value the value that is in the option
   * \returns pressio_options_key_does_not_exist if the key does not exist
   *          pressio_options_key_exists if the key exists but has no value
   *          pressio_options_key_set if the key exists and is set
   */
  template <class PointerType>
  enum pressio_options_key_status get(std::string const& key, PointerType value) const {
    using ValueType = typename std::remove_pointer<PointerType>::type;
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
   * gets a key if it is set and stores it into the pointer value
   * \param[in] name the name to use.  Checks for the named version first, then the unnamed
   * \param[in] key the option to retrieve
   * \param[out] value the value that is in the option
   * \returns pressio_options_key_does_not_exist if the key does not exist
   *          pressio_options_key_exists if the key exists but has no value
   *          pressio_options_key_set if the key exists and is set
   */
  template <class PointerType>
  enum pressio_options_key_status get(std::string const& name, std::string const& key, PointerType value) const {
    auto prefix_key = format_name(name, key);
    if(options.find(prefix_key) != options.end()) return get(prefix_key, value);
    else return get(key, value);
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
    using ValueType = typename std::remove_pointer<PointerType>::type;
    switch(key_status(key)){
      case pressio_options_key_set:
        {
          auto variant = get(key);
          auto converted = pressio_option(variant).as(pressio_type_to_enum<ValueType>(), safety);
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
   * gets a key if it is set, attepts to cast it the specified type and stores it into the pointer value
   * \param[in] name the name to use.  Checks for the named version first, then the unnamed
   * \param[in] key the option to retrieve
   * \param[out] value the value that is in the option
   * \param[in] safety the level of conversions to allow \see pressio_conversion_safety
   * \returns pressio_options_key_does_not_exist if the key does not exist
   *          pressio_options_key_exists if the key exists but has no value
   *          pressio_options_key_set if the key exists and is set
   */
  template <class PointerType>
  enum pressio_options_key_status cast(std::string const& name, std::string const& key, PointerType value, enum pressio_conversion_safety safety) const {
    auto prefix_key = format_name(name, key);
    if(options.find(prefix_key) != options.end()) return cast(prefix_key, value, safety);
    return cast(key, value, safety);
  }

  /**
   * removes all options
   */
  void clear() noexcept {
    options.clear();
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

  std::string format_name(std::string const& name, std::string const& key) const {
    if(name == "") return key;
    else return '/' + name + ':' + key;
  }

  std::map<std::string, pressio_option> options;

  public:

  /**
   * \returns an begin iterator over the keys
   */
  auto begin() const -> decltype(std::begin(options)) {
    return std::begin(options);
  }
  /**
   * \returns an end iterator over the keys
   */
  auto end() const -> decltype(std::end(options)) {
    return std::end(options);
  }

  /**
   * \returns an begin iterator over the keys
   */
  auto begin() -> decltype(std::begin(options)){
    return std::begin(options);
  }
  /**
   * \returns an end iterator over the keys
   */
  auto end() -> decltype(std::end(options)){
    return std::end(options);
  }

  /**\returns the number of set and existing options*/
  size_t size() const {
    return options.size();
  }

  /**
   * find an element of the container.  if it is not found, return end().  Useful for lua
   * \param[in] key the key to search for
   * \returns an iterator to the found key
   */
  iterator find(key_type const& key) {
    return options.find(key);
  }

  /**
   * erase a key from the container, useful for lua
   * \param[in] key the key to search for
   * \returns the number of elements erased
   */
  size_t erase(key_type const& key) {
    return options.erase(key);
  }

  /**
   * insert a new key-value pair into the options, useful for lua
   * \param[in] value the value to insert
   * \returns the number of elements erased
   */
  auto insert(value_type const& value) -> decltype(options.insert(value)) {
    return options.insert(value);
  }

  /**\returns the number of set options*/
  size_t num_set() const {
    return std::count_if(std::begin(options), std::end(options), [](decltype(options)::value_type const& key_option){
          return key_option.second.has_value();
        });
  }
};


#endif
