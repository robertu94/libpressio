#ifdef __cplusplus
extern "C" {
#endif
/*! \file
 *  \brief A single option value for a compressor
 */

#ifndef LOSSY_OPTION
/**
 * hearder guard
 */
#define LOSSY_OPTION
#include "lossy_options.h"

/** possible types contained in a lossy_option, more types may be added in the future */
enum lossy_option_type {
  /** option is a 32 bit unsigned integer */lossy_option_uint32_type=0,
  /** option is a 32 bit signed integer */lossy_option_int32_type=1,
  /** option is a 32 bit single precision floating point */lossy_option_float_type=2,
  /** option is a 64 bit double precision floating point */lossy_option_double_type=3,
  /** option is a non-owning pointer to a c-style string  */lossy_option_charptr_type=4,
  /** option is a non-owning pointer to a arbitrary data */lossy_option_userptr_type=5,
  /** option is a option that is not set */lossy_option_unset=6
};

/**
 * creates an empty lossy_option
 * \returns a new option structure with dtype lossy_option_unset
 */
struct lossy_option* lossy_option_new();
/**
 * frees the memory associated with a returned option
 * \param[in] options the option to free
 */
void lossy_option_free(struct lossy_option* options);

/**
 * gets the type of the returned lossy_option
 * \param[in] option the option to get the dtype for
 * \returns the type the option contains
 */
enum lossy_option_type lossy_option_get_type(struct lossy_option const* option);

/**
 * defines a getter and setter prototype for a lossy option type
 *
 * \param[in] name the name to append to the function
 * \param[in] type the type return or accept in the function
 *
 */
#define lossy_option_define_type(name, type) \
  /**
    Creates a new lossy_option containing the specified value
    \param[in] value the value to use to create the object \
    \returns a pointer to a new lossy option set to value passed in\
   */ \
  struct lossy_option* lossy_option_new_##name(type value); \
  /** 
    Gets the value stored in the lossy_option. Calling this with the improper dtype has undefined behavior \
    \param[in] option the option to retrieve a value from \
    \returns the value contained in the option \
   */ \
  type lossy_option_get_##name(struct lossy_option const* option); \
  /**
   Sets the option to an integer value \
   \param[in] option the option to set \
   \param[in] value the value to set   \
   */ \
  void lossy_option_set_##name(struct lossy_option* option, type value);

lossy_option_define_type(uinteger, unsigned int)
lossy_option_define_type(integer, int)
lossy_option_define_type(float, float)
lossy_option_define_type(double, double)
lossy_option_define_type(string, const char*)
lossy_option_define_type(userptr, void*)

#undef lossy_option_define_type

/**
 * \param[in] option the option to convert
 * \param[in] type the type to convert to
 * \returns a new option value of the type specified if possible, otherwise returns NULL.
 * \see lossy_option_convert behaves as if this function was called if with safety=lossy_conversion_implicit
 */
struct lossy_option* lossy_option_convert_implicit(struct lossy_option const* option, enum lossy_option_type type);
/**
 * converts between one type and another
 * \param[in] option the option to convert
 * \param[in] type the type to convert to
 * \param[in] safety how safe to make perform a conversion
 * \returns a new option value of the type specified if possible, otherwise returns NULL.
 */
struct lossy_option* lossy_option_convert(struct lossy_option const* option, enum lossy_option_type type, enum lossy_conversion_safety safety);

#endif

#ifdef __cplusplus
}
#endif
