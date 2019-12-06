#ifdef __cplusplus
extern "C" {
#endif
/*! \file
 *  \brief A single option value for a compressor
 */

#ifndef PRESSIO_OPTION
/**
 * hearder guard
 */
#define PRESSIO_OPTION
#include <stdbool.h>
#include "pressio_options.h"


/**
 * creates an empty pressio_option
 * \returns a new option structure with dtype pressio_option_unset
 */
struct pressio_option* pressio_option_new();
/**
 * frees the memory associated with a returned option
 * \param[in] options the option to free
 */
void pressio_option_free(struct pressio_option* options);

/**
 * gets the type of the returned pressio_option
 * \param[in] option the option to get the dtype for
 * \returns the type the option contains
 */
enum pressio_option_type pressio_option_get_type(struct pressio_option const* option);

/**
 * set the type of the passed pressio_option
 * \param[in] option the option to get the dtype for
 * \param[in] type the option to get the dtype for
 */
void pressio_option_set_type(struct pressio_option* option, enum pressio_option_type type);

/**
 * returns true if the option contains a value
 * \param[in] option the option to check for a value
 * \return true if the option contains a value
 */
bool pressio_option_has_value(struct pressio_option const* option);

/**
 * Sets pressio_option for the specified key with a generic pressio_options preserving the type of the key in the options structure
 * using the specified cast if necessary
 * \param[in] lhs the option to get an pressio_option for
 * \param[in] rhs value to assign in the pressio_options structure
 * \param[in] safety what kind of conversions to allow
 * \returns pressio_options_key_set if the lhs is modified, pressio_options_key_exists otherwise
 */
enum pressio_options_key_status pressio_option_cast_set(struct pressio_option* lhs, struct pressio_option* rhs, enum pressio_conversion_safety safety);

/**
 * Sets pressio_option for the specified key with a generic pressio_options preserving the type of the key in the options structure
 * using an implicit cast if necessary
 * \param[in] lhs the option to get an pressio_option for
 * \param[in] rhs value to assign in the pressio_options structure
 * \returns pressio_options_key_set if the lhs is modified, pressio_options_key_exists otherwise
 */
enum pressio_options_key_status pressio_option_as_set(struct pressio_option* lhs, struct pressio_option* rhs);

/**
 * defines a getter and setter prototype for a pressio option type
 *
 * \param[in] name the name to append to the function
 * \param[in] type the type return or accept in the function
 *
 */
#define pressio_option_define_type(name, type) \
  /**
    Creates a new pressio_option containing the specified value
    \param[in] value the value to use to create the object \
    \returns a pointer to a new pressio option set to value passed in\
   */ \
  struct pressio_option* pressio_option_new_##name(type value); \
  /** 
    Gets the value stored in the pressio_option. Calling this with the improper dtype has undefined behavior \
    \param[in] option the option to retrieve a value from \
    \returns the value contained in the option \
   */ \
  type pressio_option_get_##name(struct pressio_option const* option); \
  /**
   Sets the option to an integer value \
   \param[in] option the option to set \
   \param[in] value the value to set   \
   */ \
  void pressio_option_set_##name(struct pressio_option* option, type value);

pressio_option_define_type(uinteger, unsigned int)
pressio_option_define_type(integer, int)
pressio_option_define_type(float, float)
pressio_option_define_type(double, double)
pressio_option_define_type(string, const char*)
pressio_option_define_type(userptr, void*)

#undef pressio_option_define_type

/**
 * \param[in] option the option to convert
 * \param[in] type the type to convert to
 * \returns a new option value of the type specified if possible, otherwise returns NULL.
 * \see pressio_option_convert behaves as if this function was called if with safety=pressio_conversion_implicit
 */
struct pressio_option* pressio_option_convert_implicit(struct pressio_option const* option, enum pressio_option_type type);
/**
 * converts between one type and another
 * \param[in] option the option to convert
 * \param[in] type the type to convert to
 * \param[in] safety how safe to make perform a conversion
 * \returns a new option value of the type specified if possible, otherwise returns NULL.
 */
struct pressio_option* pressio_option_convert(struct pressio_option const* option, enum pressio_option_type type, enum pressio_conversion_safety safety);

#endif

#ifdef __cplusplus
}
#endif
