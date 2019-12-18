#ifdef __cplusplus
extern "C" {
#endif


#ifndef LIBPRESSIO_OPTIONS_H
/**
 * Header Guard
 */
#define LIBPRESSIO_OPTIONS_H
#include "pressio_dtype.h"

/*! \file  
 *  \brief A set of options for a compressor
 */

struct pressio_options;
struct pressio_options_iter;
struct pressio_option;

/** possible status of a particular key in the option structure*/
enum  pressio_options_key_status{
  /** the requested key exists and is set, evaluates to false*/
  pressio_options_key_set=0,

  /** the requested key exists but is not set, for pressio_option_set_* functions indicates a type mismatch*/
  pressio_options_key_exists=1,

  /** the requested key does not exist */
  pressio_options_key_does_not_exist=2,
};

/** level of safety to require for conversions*/
enum pressio_conversion_safety {
  /** conversions that are implicitly convertible without compiler warnings in
   * C++
   */
  pressio_conversion_implicit=0,

  /** all of the above, and conversions that are explicitly convertible with a
   * cast in C++
   */
  pressio_conversion_explicit=1,

  /** all of the above, and conversions that require a special function call
   * (i.e. atoi) if this function fails, NULL will be returned
   */
  pressio_conversion_special=2,
};

/** possible types contained in a pressio_option, more types may be added in the future */
enum pressio_option_type {
  /** option is a 32 bit unsigned integer */pressio_option_uint32_type=0,
  /** option is a 32 bit signed integer */pressio_option_int32_type=1,
  /** option is a 32 bit single precision floating point */pressio_option_float_type=2,
  /** option is a 64 bit double precision floating point */pressio_option_double_type=3,
  /** option is a non-owning pointer to a c-style string  */pressio_option_charptr_type=4,
  /** option is a non-owning pointer to a arbitrary data */pressio_option_userptr_type=5,
  /** option is a non-owning pointer to a arbitrary data */pressio_option_unset_type=6,
};


/**
 * Creates an empty pressio_options structure returns NULL if the allocation fails
 * \returns a pointer to the new options structure
 * \see pressio_options_free 
 */
struct pressio_options* pressio_options_new();
/**
 * Copies the memory associated with this pressio option structure
 *
 * \param[in] options the options structure to copy
 * \returns a pointer to the copied options structure
 */
struct pressio_options* pressio_options_copy(struct pressio_options const* options);
/**
 * Merges two pressio options together into one.  Copies all keys and
 * corresponding values from rhs not in lhs into a new structure.
 * 
 * \param[in] lhs the first structure to merge
 * \param[in] rhs the second structure to merge; if lhs and rhs both have the same key the value from lhs is preserved
 * \return a newly allocated pressio_options structure.
 */
struct pressio_options* pressio_options_merge(struct pressio_options const* lhs, struct pressio_options const* rhs);
/**
 * Frees the memory associated with a pressio option structure
 *
 * \param[in,out] options frees the pressio option structure
 */
void pressio_options_free(struct pressio_options* options);

/**
 * \param[in] options the option to get an pressio_options_key_status for
 * \param[in] key the key to get from the options structure
 * \returns a key status for the requested key
 */
enum pressio_options_key_status pressio_options_exists(struct pressio_options const* options, const char* key);


/**
 * Gets a generic pressio_option for the specified key.  Calling this with an nonexistent key has undefined behavior
 * \param[in] options the option to get an pressio_option for
 * \param[in] key the key to get from the options structure
 * \returns a new copy of the corresponding pressio_option
 */
struct pressio_option* pressio_options_get(struct pressio_options const* options, const char* key);
/**
 * Sets pressio_option for the specified key with a generic pressio_options
 * \param[in] options the option to get an pressio_option for
 * \param[in] key the key to get from the options structure
 * \param[in] option value to set in the pressio_options structure
 */
void pressio_options_set(struct pressio_options* options, const char* key, struct pressio_option* option);

/**
 * Sets pressio_option for the specified key with a generic pressio_options preserving the type of the key in the options structure
 * using the specified cast if necessary
 * \param[in] options the option to get an pressio_option for
 * \param[in] key the key to get from the options structure
 * \param[in] option value to assign in the pressio_options structure
 * \param[in] safety what kind of conversions to allow
 * \returns pressio_options_key_set if the lhs is modified, pressio_options_key_exists if the key exists but can't convert, and pressio_options_key_does_not_exist if the key does not exist;
 */
enum pressio_options_key_status pressio_options_cast_set(struct pressio_options* options, const char* key, struct pressio_option* option, enum pressio_conversion_safety safety);

/**
 * Sets pressio_option for the specified key with a generic pressio_options preserving the type of the key in the options structure
 * using an implicit cast if necessary
 * \param[in] options the option to get an pressio_option for
 * \param[in] key the key to get from the options structure
 * \param[in] option value to assign in the pressio_options structure
 * \returns pressio_options_key_set if the lhs is modified, pressio_options_key_exists if the key exists but can't convert, and pressio_options_key_does_not_exist if the key does not exist;
 */
enum pressio_options_key_status pressio_options_as_set(struct pressio_options* options, const char* key, struct pressio_option* option);

/**
 * Sets pressio_option to the specified type
 * \param[in] options the option to set type of a pressio_option for
 * \param[in] key the key to get from the options structure
 * \param[in] type value to set in the pressio_options structure
 * \returns pressio_options_key_set if the key is modified, pressio_options_key_exists if the key exists but can't convert, and pressio_options_key_does_not_exist if the key does not exist;
 */
void pressio_options_set_type(struct pressio_options* options, const char* key, enum pressio_option_type type);

/**
 * Clear the value associated with a key, but retains the entry in the options.
 *
 * This MAY be used by libpressio plugin implementations to suggest a key to set
 * libpressio plugin implementations SHOULD not change the underlying setting if the option has been cleared
 * 
 * \param[in] options the options structure to clear a value for
 * \param[in] key the key whose value to clear
 */
void pressio_options_clear(struct pressio_options* options, const char* key);


/** internal macro used to define setter functions */
#define pressio_options_define_type_set(name, type) \
  /** Sets an particular key in an options structure with the given key to a value
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[in] value the value to change to
   */ \
  void pressio_options_set_##name(struct pressio_options* options, const char* key, type value);

/** internal macro used to define getter functions */
#define pressio_options_define_type_get(name, type) \
  /** Gets a particular value in a map if it exists
   *
   * pressio_options_get_string returns a newly allocated copy of the string
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[out] value the value retrieved
   \returns a status code
   \see pressio_options_key_status status codes returned
   */ \
  enum pressio_options_key_status pressio_options_get_##name(struct pressio_options \
      const* options, const char* key, type * value);

/** internal macro used to define casting functions */
#define pressio_options_define_type_cast(name, type) \
  /** Gets an particular key in an options structure, casting it if necessary
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[in] safety  what kind of conversions to allow
   \param[out] value the value retrieved, only if it is convertible. If returning a char*, the memory must be freed with free()
   \returns a status code 
   \see pressio_options_key_status status codes returned 
   */ \
  enum pressio_options_key_status pressio_options_cast_##name(struct pressio_options \
      const* options, const char* key, const enum pressio_conversion_safety safety, \
      type * value);

/** internal macro used to define implicit casting functions */
#define pressio_options_define_type_as(name, type) \
  /** Gets an particular key in an options structure, casting it if necessary
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[out] value the value retrieved, only if it is convertible. If returning a char*, the memory must be freed with free()
   \returns a status code
   \see pressio_options_key_status status codes returned
   */ \
  enum pressio_options_key_status pressio_options_as_##name(struct pressio_options \
      const* options, const char* key, type * value);

/**
 * Generate get/set/as/cast functions for the pressio_options class
 * \param[in] name the name to append to the function
 * \param[in] type the type to wrap
 */
#define pressio_options_define_type(name, type) \
  pressio_options_define_type_set(name, type) \
  pressio_options_define_type_get(name, type) \
  pressio_options_define_type_cast(name, type) \
  pressio_options_define_type_as(name, type) 

pressio_options_define_type(uinteger, unsigned int)
pressio_options_define_type(integer, int)
pressio_options_define_type(float, float)
pressio_options_define_type(double, double)
pressio_options_define_type(userptr, void*)

//special case string
pressio_options_define_type_get(string, const char*)
pressio_options_define_type_set(string, const char*)
pressio_options_define_type_as(string, char*)
pressio_options_define_type_cast(string, char*)

//undefine the type macro so it is not used by implementations
#undef pressio_options_define_type
#undef pressio_options_define_type_set
#undef pressio_options_define_type_get
#undef pressio_options_define_type_as
#undef pressio_options_define_type_cast

#endif

#ifdef __cplusplus
}
#endif
