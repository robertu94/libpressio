#ifndef LIBPRESSIO_REGISTRY_H
#define LIBPRESSIO_REGISTRY_H
#include <string>
#include <map>
#include <functional>

/**
 * \file
 * \brief C++ Interface for registries of construct-able objects
 */

namespace libpressio {

/**
 * a type that registers constructor functions
 */
template <class T>
struct pressio_registry {
  /**
   * construct a element of the registered type
   *
   * \param[in] name the item to construct
   * \returns the result of the factory function
   */
  T build(std::string const& name) const {
    auto factory = factories.find(name);
    if ( factory != factories.end()) {
      return factory->second();
    } else {
      return nullptr;
    }
  }

  /**
   * register a factory function with the registry at the provided name
   *
   * \param[in] name the name to register
   * \param[in] factory the constructor function which takes 0 arguments
   * \return true if the module has already not been registered
   */
  template <class Name, class Factory>
  bool regsiter_factory(Name&& name, Factory&& factory) {
    std::string name_copy(name);
    if (factories.find(name) == factories.end()) {
        factories.emplace(std::move(name_copy), std::forward<Factory>(factory));
        return false;
    } else {
        return true;
    }
  }

  private:
  std::map<std::string, std::function<T()>> factories;

  public:
  /** the value type the registry constructs*/
  using value_type = T;
  /** the reference type the registry constructs*/
  using reference = T&;
  /** the const reference type the registry constructs*/
  using const_reference = T const;
  /**
   * \returns an begin iterator over the registered types
   */
  auto begin() const -> decltype(factories.begin()) { return std::begin(factories); }
  /**
   * \returns an end iterator over the registered types
   */
  auto end() const -> decltype(factories.end()) { return std::end(factories); }

  /**
   * remove an entry from the factory if it exists
   * \returns 1 if an item was removed, 0 otherwise
   */
  size_t erase(std::string const& name) {
      return factories.erase(name);
  }

  /**
   * checks if the name is registered
   *
   * \param[in] key the key to search for
   * \returns true if present
   */
  bool contains(std::string const& key) const {
    return factories.find(key) != factories.end();
  }

  /**
   * checks if the name is registered
   *
   * \param[in] key the key to search for
   * \returns an iterator if the entry is found; else end()
   */
  auto find(std::string const& key) const -> decltype(factories.find(key)) {
    return factories.find(key);
  }

  /**
   * checks if the name is registered
   *
   * \param[in] key the key to search for
   * \returns an iterator if the entry is found; else end()
   */
  auto find(std::string const& key) -> decltype(factories.find(key)) {
    return factories.find(key);
  }

  /**
   * return the number of entries in the registry
   */
  size_t size() const {
      return factories.size();
  }

};

/**
 * a class that registers a type on construction, using a type over a function
 * to force it to be called at static construction time
 */
class pressio_register{
  public:
  /**
   * Registers a new factory with a name in a registry.  Designed to be used as a static variable
   *
   * \param[in] registry the registry to use
   * \param[in] name the name to register
   * \param[in] factory the factory to register
   */
  template <class RegistryType, class NameType, class Factory>
  pressio_register(pressio_registry<RegistryType>& registry, NameType&& name, Factory factory):
      name(name),
      unregister([&registry, name]{ registry.erase(name);}),
      do_register([&registry, factory, name]{ return registry.regsiter_factory(name, factory);})
    {
    do_register();
  }

  ~pressio_register(){
      unregister();
  }

  bool ensure_registered() {
      return do_register();
  }

  /** the name of the registered plugin */
  std::string name;
  /** call to unregister the plugin */
  std::function<void()> unregister;
  std::function<bool()> do_register;
};

}

#endif /* end of include guard: LIBPRESSIO_REGISTRY_H */
