#ifndef LIBPRESSIO_CONFIGURABLE_H
#define LIBPRESSIO_CONFIGURABLE_H
#include <string>
#include <utility>
#include "options.h"
#include "pressio_compressor.h"

/**
 * \file
 * \brief interface for configurable types
 */

/**
 * Base interface for configurable objects in libpressio
 */
class pressio_configurable {
  public:
  virtual ~pressio_configurable()=default;
  pressio_configurable()=default;

  /** get the prefix used by this compressor for options */
  virtual const char* prefix() const=0;

  /**
   * \returns the assigned name for the compressor used in options getting/setting
   */
  std::string const& get_name() const {
    return name;
  }

  /**
   * sets the assigned name for the compressor used in options getting/setting
   * \param[in] new_name the name to be used
   */
  virtual void set_name(std::string const& new_name) {
    this->set_name_impl(new_name);
    this->name = new_name;
  }

  /**
   * Meta-compressors need to know when names are changed so they can update their children
   *
   * \param[in] new_name the name to be used
   */
  virtual void set_name_impl(std::string const& new_name) {
    (void)new_name;
  }

  /** get the compile time configuration of a configurable object
   *
   * \see pressio_compressor_get_configuration for the semantics this function should obey
   */
  virtual struct pressio_options get_configuration() const;

  /** get the documentation strings for a compressor
   *
   * \see pressio_compressor_get_documentation for the semantics this function should obey
   */
  virtual struct pressio_options get_documentation() const;

  /** checks for extra arguments set for the configurable object.
   *
   * \see pressio_compressor_check_options for semantics this function obeys
   * */
  virtual int check_options(struct pressio_options const&);

  /** get a set of options available for the configurable object.
   *
   * The compressor should set a value if they have been set as default
   * The compressor should set a "reset" value if they are required to be set, but don't have a meaningful default
   *
   * \see pressio_compressor_get_options for the semantics this function should obey
   * \see pressio_options_clear to set a "reset" value
   * \see pressio_options_set_integer to set an integer value
   * \see pressio_options_set_double to set an double value
   * \see pressio_options_set_userptr to set an data value, include an \c include/ext/\<my_plugin\>.h to define the structure used
   * \see pressio_options_set_string to set a string value
   */
  virtual struct pressio_options get_options() const;

  /** sets a set of options for the configurable object 
   * \param[in] options to set for configuration of the configurable object
   * \see pressio_compressor_set_options for the semantics this function should obey
   */
  virtual int set_options(struct pressio_options const& options);

  protected:
  /**
   * helper function to set options according to name prefixes if provided
   *
   * \param[in] options the options structure to set
   * \param[in] key the key to set
   * \param[in] value the value to set
   */
  template <class StringType>
  void set(pressio_options& options, StringType const& key, pressio_option const& value) const {
    if(name.empty()) options.set(key, value);
    else options.set(name, key, value);
  }


  /**
   * helper function to set the type of options according to name prefixes if provided
   *
   * \param[in] options the options structure to set
   * \param[in] key the key to set
   * \param[in] type the type to set
   */
  template <class StringType>
  void set_type(pressio_options& options, StringType const& key, pressio_option_type type) const {
    if(name.empty()) options.set_type(key, type);
    else options.set_type(name, key, type);
  }


  /**
   * helper function to get the type of options according to name prefixes if provided
   *
   * \param[in] options the options structure to set
   * \param[in] key the key to set
   * \param[in] value the value to get
   * \returns if the key was set
   */
  template <class StringType, class PointerType>
  enum pressio_options_key_status 
  get(pressio_options const& options, StringType && key, PointerType value) const {
    if(name.empty()) return options.get(std::forward<StringType>(key), value); 
    return options.get(name, std::forward<StringType>(key), value);
  }

  /**
   * helper function to set on a pressio_options structure the values associated with a meta-object
   *
   * \param[in] options the options structure to set
   * \param[in] key the key for the name of the child meta-object
   * \param[in] current_id name for the current meta-object
   * \param[in] current_value value of the current meta-object
   * \param[in] args the remaining args needed for some meta modules
   */
  template<class StringType, class Wrapper, class... Args>
  void
  set_meta(pressio_options& options, StringType&& key, std::string const& current_id, Wrapper const& current_value, Args&&... args) const {
    set(options, std::forward<StringType>(key), current_id);
    options.copy_from(current_value->get_options(std::forward<Args>(args)...));
  }

  /**
   * helper function to set docs on a pressio_options structure the values associated with a meta-object
   *
   * \param[in] options the options structure to set
   * \param[in] key the key for the name of the child meta-object
   * \param[in] docstring docs for the purpose of the meta object
   * \param[in] current_value value of the current meta-object
   * \param[in] args the remaining args needed for some meta modules
   */
  template<class StringType, class Wrapper, class... Args>
  void
  set_meta_docs(pressio_options& options, StringType&& key, std::string const& docstring, Wrapper const& current_value, Args&&... args) const {
    set(options, std::forward<StringType>(key), docstring);
    options.copy_from(current_value->get_documentation(std::forward<Args>(args)...));
  }

  /**
   * helper function to set on a pressio_options structure the docs associated with a collection of meta-objects
   *
   * \param[in] options the options structure to set
   * \param[in] key the key for the name of the child meta-object
   * \param[in] docstring docs for the purpose of this collection of meta objects
   * \param[in] current_values value of the current meta-object
   * \param[in] args the remaining args needed for some meta modules
   */
  template<class StringType, class Wrapper, class... Args>
  void
  set_meta_many_docs(pressio_options& options, StringType&& key, std::string const& docstring, std::vector<Wrapper> const& current_values, Args&&... args) const {
    set(options, std::forward<StringType>(key), docstring);
    for (auto const& wrapper : current_values) {
      options.copy_from(wrapper->get_documentation(std::forward<Args>(args)...));
    }
  }

  /**
   * helper function to set on a pressio_options structure the values associated with a collection of meta-objects
   *
   * \param[in] options the options structure to set
   * \param[in] key the key for the name of the child meta-object
   * \param[in] current_ids name for the current meta-object
   * \param[in] current_values value of the current meta-object
   * \param[in] args the remaining args needed for some meta modules
   */
  template<class StringType, class Wrapper, class... Args>
  void
  set_meta_many(pressio_options& options, StringType&& key, std::vector<std::string> const& current_ids, std::vector<Wrapper> const& current_values, Args&&... args) const {
    set(options, std::forward<StringType>(key), current_ids);
    for (auto const& wrapper : current_values) {
      options.copy_from(wrapper->get_options(std::forward<Args>(args)...));
    }
  }

  /**
   * helper function to get from a pressio_options structure the values associated with a meta-object
   *
   * \param[in] options the options to set
   * \param[in] key the meta-key to query
   * \param[in] registry the registry to construct new values from
   * \param[in] current_id the id of the current module
   * \param[in] current_value the wrapper for the current module
   *
   */
  template <class StringType, class Registry, class Wrapper>
  pressio_options_key_status
  get_meta(pressio_options const& options,
      StringType && key,
      Registry const& registry,
      std::string& current_id,
      Wrapper& current_value) {
    std::string new_id;
    if(get(options, std::forward<StringType>(key), &new_id) == pressio_options_key_set) {
      if (new_id != current_id) {
        auto new_value = registry.build(new_id);
        if(new_value) {
          current_id = std::move(new_id);
          current_value = std::move(new_value);
        } else {
          return pressio_options_key_does_not_exist;
        }
        if(not get_name().empty()) {
          set_name(get_name());
        }
      }
    }
    current_value->set_options(options);
    return pressio_options_key_exists;
  }

  /**
   * helper function to get from a pressio_options structure the values associated with multiple meta-objects
   *
   * \param[in] options the options to set
   * \param[in] key the meta-key to query
   * \param[in] registry the registry to construct new values from
   * \param[in] current_ids the id of the current module
   * \param[in] current_values the wrapper for the current module
   *
   */
  template <class StringType, class Registry, class Wrapper>
  pressio_options_key_status
  get_meta_many(pressio_options const& options,
      StringType && key,
      Registry const& registry,
      std::vector<std::string>& current_ids,
      std::vector<Wrapper>& current_values) {
    std::vector<std::string> new_ids;
    if(get(options, std::forward<StringType>(key), &new_ids) == pressio_options_key_set) {
      if (new_ids != current_ids) {
        std::vector<Wrapper> new_values;
        bool all_built = true;
        for (auto const& new_id : new_ids) {
          new_values.emplace_back(registry.build(new_id));
          all_built &= new_values.back();
          if(not all_built) break;
        }
        if(all_built) {
          current_ids = std::move(new_ids);
          current_values = std::move(new_values);
        } else {
          return pressio_options_key_does_not_exist;
        }
        if(not get_name().empty()) {
          set_name(get_name());
        }
      }
    }
    for (auto& current_value : current_values) {
      current_value->set_options(options);
    }
    return pressio_options_key_exists;
  }


  protected:

  /**
   * 
   * \returns that returns the thread_safe configuration parameter
   */
  static pressio_thread_safety get_threadsafe(pressio_configurable const& c) {
    int32_t thread_safe = pressio_thread_safety_single;
    c.get_configuration().get("pressio:thread_safe", &thread_safe);
    return pressio_thread_safety(thread_safe);
  }

  /**
   * \returns the string the metrics key name
   * \internal
   */
  std::string get_metrics_key_name() const {
    return std::string(prefix()) + ":metric";
  }

  /** the name of the configurable used in nested hierarchies*/
  std::string name;
};

#endif /* end of include guard: LIBPRESSIO_CONFIGURABLE_H */
