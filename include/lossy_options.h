#ifdef __cplusplus
extern "C" {
#endif


#ifndef LIBLOSSY_OPTIONS_H
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

/**
 * Frees the memory associated with a lossy option structure
 *
 * \param[in,out] options frees the lossy option structure
 */
void lossy_options_free(struct lossy_options* options);

/**
 * Copies the memory associated with this lossy option structure
 *
 * \param[in] options the options structure to copy
 * \returns a pointer to the copied options structure
 */
struct lossy_options* lossy_options_copy(struct lossy_options const* options);

/**
 * Creates an empty lossy_options structure returns NULL if the allocation fails
 * \returns a pointer to the new options structure
 * \see lossy_options_free 
 */
struct lossy_options* lossy_options_new();

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

/**
 * Merges two lossy options together into one.  Copies all keys and
 * corresponding values from rhs not in lhs into a new structure.
 * 
 * \param[in] lhs the structure to insert keys into.
 * \param[in] rhs the lossy_options structure to merge in.  It is deallocated when the function returns.
 * \return a new lossy_options structure.
 */
struct lossy_options* lossy_options_merge(struct lossy_options const* lhs, struct lossy_options const* rhs);

/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[in] value the value to change to
 */
void lossy_options_set_userptr(struct lossy_options* options, const char* key, void* value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[in] value the value to change to
 */
void lossy_options_set_uinteger(struct lossy_options* options, const char* key, unsigned int value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[in] value the value to change to
 */
void lossy_options_set_integer(struct lossy_options* options, const char* key, int value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[in] value the value to change to
 */
void lossy_options_set_string(struct lossy_options* options, const char* key, const char* value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[in] value the value to change to
 */
void lossy_options_set_float(struct lossy_options* options, const char* key, float value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[in] value the value to change to
 */
void lossy_options_set_double(struct lossy_options* options, const char* key, double value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[out] value the value retrieved
 *  \returns a status code
 *  \see lossy_options_key_status status codes returned
 */
enum lossy_options_key_status lossy_options_get_userptr(struct lossy_options const* options, const char* key, void** value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[out] value the value retrieved, only 
 *  \returns a status code
 *  \see lossy_options_key_status status codes returned
 */
enum lossy_options_key_status lossy_options_get_uinteger(struct lossy_options const* options, const char* key, unsigned int* value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[out] value the value retrieved, only 
 *  \returns a status code
 *  \see lossy_options_key_status status codes returned
 */
enum lossy_options_key_status lossy_options_get_integer(struct lossy_options const* options, const char* key, int* value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[out] value the value retrieved
 *  \returns a status code
 *  \see lossy_options_key_status status codes returned
 */
enum lossy_options_key_status lossy_options_get_string(struct lossy_options const* options, const char* key, const char** value);
/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[out] value the value retrieved
 *  \returns a status code
 *  \see lossy_options_key_status status codes returned
 */
enum lossy_options_key_status lossy_options_get_float(struct lossy_options const* options, const char* key, float* value);

/** Sets an particular key in an options structure with the given key to a value 
 *  \param[in] options the options structure to modify 
 *  \param[in] key  the key to change
 *  \param[out] value the value retrieved
 *  \returns a status code
 *  \see lossy_options_key_status status codes returned
 */
enum lossy_options_key_status lossy_options_get_double(struct lossy_options const* options, const char* key, double* value);


/**
 * \param[in] options the option to get an lossy_options_key_status for
 * \param[in] key the key to get from the options structure
 * \returns a key status for the requested key
 */
enum lossy_options_key_status lossy_options_exists(struct lossy_options const* options, const char* key);

/**
 * sets an option with the value passed to value
 * \param[in] options the option to get an lossy_option for
 * \param[in] key the key to get from the options structure
 * \returns the corresponding lossy_option
 */
struct lossy_option* lossy_options_get(struct lossy_options const* options, const char* key);




#endif

#ifdef __cplusplus
}
#endif
