#ifdef __cplusplus
extern "C" {
#endif
/*! \file
 *  \brief A single option value for a compressor
 */



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
 * Gets the value stored in the lossy_option. Calling this with the improper dtype has undefined behavior
 * \param[in] option the option to retrieve a value from
 * \returns the value contained in the option
 */
int lossy_option_get_integer(struct lossy_option const* option);
/**
 * Gets the value stored in the lossy_option. Calling this with the improper dtype has undefined behavior
 * \param[in] option the option to retrieve a value from
 * \returns the value contained in the option
 */
unsigned int lossy_option_get_uinteger(struct lossy_option const* option);
/**
 * Gets the value stored in the lossy_option. Calling this with the improper dtype has undefined behavior
 * \param[in] option the option to retrieve a value from
 * \returns the value contained in the option
 */
float lossy_option_get_float(struct lossy_option const* option);
/**
 * Gets the value stored in the lossy_option. Calling this with the improper dtype has undefined behavior
 * \param[in] option the option to retrieve a value from
 * \returns the value contained in the option
 */
double lossy_option_get_double(struct lossy_option const* option);
/**
 * Gets the value stored in the lossy_option. Calling this with the improper dtype has undefined behavior
 * \param[in] option the option to retrieve a value from
 * \returns non-owning pointer to the value contained in the option
 */
const char* lossy_option_get_string(struct lossy_option const* option);
/**
 * Gets the value stored in the lossy_option. Calling this with the improper dtype has undefined behavior
 * \param[in] option the option to retrieve a value from
 * \returns non-owning pointer to the value contained in the option
 */
void* lossy_option_get_userptr(struct lossy_option const* option);

/**
 * Sets the option to an integer value
 * \param[in] option the option to set
 * \param[in] value the value to set
 */
void lossy_option_set_uinteger(struct lossy_option* option, unsigned int value);
/**
 * Sets the option to an integer value
 * \param[in] option the option to set
 * \param[in] value the value to set
 */
void lossy_option_set_integer(struct lossy_option* option, int value);
/**
 * Sets the option to an double value
 * \param[in] option the option to set
 * \param[in] value the value to set
 */
void lossy_option_set_float(struct lossy_option* option, float value);
/**
 * Sets the option to an double value
 * \param[in] option the option to set
 * \param[in] value the value to set
 */
void lossy_option_set_double(struct lossy_option* option, double value);
/**
 * Sets the option to an c-string value
 * \param[in] option the option to set
 * \param[in] value the value to set
 */
void lossy_option_set_string(struct lossy_option* option, const char* value);
/**
 * Sets the option to an non-owning void* value
 * \param[in] option the option to set
 * \param[in] value the value to set
 */
void lossy_option_set_userptr(struct lossy_option* option, void* value);

#ifdef __cplusplus
}
#endif
