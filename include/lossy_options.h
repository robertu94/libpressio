#ifdef __cplusplus
extern "C" {
#endif


#ifndef LIBLOSSY_OPTIONS_H
/**
 * Header Guard
 */
#define LIBLOSSY_OPTIONS_H
#include "lossy_dtype.h"

/*! \file  
 *  \brief A set of options for a compressor
 */

struct lossy_options;
struct lossy_options_iter;
struct lossy_option;

/** possible status of a particular key in the option structure*/
enum  lossy_options_key_status{
  /** the requested key exists and is set*/
  lossy_options_key_set=0,

  /** the requested key exists but is not set, for lossy_option_set_* functions indicates a type mismatch*/
  lossy_options_key_exists=1,

  /** the requested key does not exist */
  lossy_options_key_does_not_exist=2,
};

/** level of safety to require for conversions*/
enum lossy_conversion_safety {
  /** conversions that are implicitly convertible without compiler warnings in
   * C++
   */
  lossy_conversion_implicit=0,

  /** all of the above, and conversions that are explicitly convertible with a
   * cast in C++
   */
  lossy_conversion_explicit=1,

  /** all of the above, and conversions that require a special function call
   * (i.e. atoi) if this function fails, NULL will be returned
   */
  lossy_conversion_special=2,
};


/**
 * Creates an empty lossy_options structure returns NULL if the allocation fails
 * \returns a pointer to the new options structure
 * \see lossy_options_free 
 */
struct lossy_options* lossy_options_new();
/**
 * Copies the memory associated with this lossy option structure
 *
 * \param[in] options the options structure to copy
 * \returns a pointer to the copied options structure
 */
struct lossy_options* lossy_options_copy(struct lossy_options const* options);
/**
 * Merges two lossy options together into one.  Copies all keys and
 * corresponding values from rhs not in lhs into a new structure.
 * 
 * \param[in] lhs the structure to insert keys into.
 * \param[in] rhs the lossy_options structure to merge in.  It is deallocated when the function returns.
 * \return a new lossy_options structure.
 */
struct lossy_options* lossy_options_merge(struct lossy_options const* lhs, struct lossy_options const* rhs);
/**
 * Frees the memory associated with a lossy option structure
 *
 * \param[in,out] options frees the lossy option structure
 */
void lossy_options_free(struct lossy_options* options);

/**
 * \param[in] options the option to get an lossy_options_key_status for
 * \param[in] key the key to get from the options structure
 * \returns a key status for the requested key
 */
enum lossy_options_key_status lossy_options_exists(struct lossy_options const* options, const char* key);


/**
 * Gets a generic lossy_option for the specified key.  Calling this with an nonexistent key has undefined behavior
 * \param[in] options the option to get an lossy_option for
 * \param[in] key the key to get from the options structure
 * \returns a new copy of the corresponding lossy_option
 */
struct lossy_option* lossy_options_get(struct lossy_options const* options, const char* key);
/**
 * Sets lossy_option for the specified key with a generic lossy_options
 * \param[in] options the option to get an lossy_option for
 * \param[in] key the key to get from the options structure
 * \param[in] option value to set in the lossy_options structure
 * \returns a new copy of the corresponding lossy_option
 */
struct lossy_option* lossy_options_set(struct lossy_options* options, const char* key, struct lossy_option* option);

/**
 * Clear the value associated with a key, but retains the entry in the options.
 *
 * This MAY be used by liblossy plugin implementations to suggest a key to set
 * liblossy plugin implementations SHOULD not change the underlying setting if the option has been cleared
 * 
 * \param[in] options the options structure to clear a value for
 * \param[in] key the key whose value to clear
 */
void lossy_options_clear(struct lossy_options* options, const char* key);


/** internal macro used to define setter functions */
#define lossy_options_define_type_set(name, type) \
  /** Sets an particular key in an options structure with the given key to a value
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[in] value the value to change to
   */ \
  void lossy_options_set_##name(struct lossy_options* options, const char* key, type value);

/** internal macro used to define getter functions */
#define lossy_options_define_type_get(name, type) \
  /** Gets a particular value in a map if it exists
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[out] value the value retrieved
   \returns a status code
   \see lossy_options_key_status status codes returned
   */ \
  enum lossy_options_key_status lossy_options_get_##name(struct lossy_options \
      const* options, const char* key, type * value);

/** internal macro used to define casting functions */
#define lossy_options_define_type_cast(name, type) \
  /** Gets an particular key in an options structure, casting it if necessary
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[in] safety  what kind of conversions to allow
   \param[out] value the value retrieved, only if it is convertible. If returning a char*, the memory must be freed with free()
   \returns a status code 
   \see lossy_options_key_status status codes returned 
   */ \
  enum lossy_options_key_status lossy_options_cast_##name(struct lossy_options \
      const* options, const char* key, const enum lossy_conversion_safety safety, \
      type * value);

/** internal macro used to define implicit casting functions */
#define lossy_options_define_type_as(name, type) \
  /** Gets an particular key in an options structure, casting it if necessary
   \param[in] options the options structure to modify
   \param[in] key  the key to change
   \param[out] value the value retrieved, only if it is convertible. If returning a char*, the memory must be freed with free()
   \returns a status code
   \see lossy_options_key_status status codes returned
   */ \
  enum lossy_options_key_status lossy_options_as_##name(struct lossy_options \
      const* options, const char* key, type * value);

/**
 * Generate get/set/as/cast functions for the lossy_options class
 * \param[in] name the name to append to the function
 * \param[in] type the type to wrap
 */
#define lossy_options_define_type(name, type) \
  lossy_options_define_type_set(name, type) \
  lossy_options_define_type_get(name, type) \
  lossy_options_define_type_cast(name, type) \
  lossy_options_define_type_as(name, type) 

lossy_options_define_type(uinteger, unsigned int)
lossy_options_define_type(integer, int)
lossy_options_define_type(float, float)
lossy_options_define_type(double, double)
lossy_options_define_type(userptr, void*)

//special case string
lossy_options_define_type_get(string, const char*)
lossy_options_define_type_set(string, const char*)
lossy_options_define_type_as(string, char*)
lossy_options_define_type_cast(string, char*)

//undefine the type macro so it is not used by implementations
#undef lossy_options_define_type
#undef lossy_options_define_type_set
#undef lossy_options_define_type_get
#undef lossy_options_define_type_as
#undef lossy_options_define_type_cast

#endif

#ifdef __cplusplus
}
#endif
