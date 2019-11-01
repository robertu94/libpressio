#ifdef __cplusplus 
extern "C" {
#endif 

#ifndef LIBPRESSIO_H
#define LIBPRESSIO_H

/*! \file
 * \brief Pressio compressor loader
 */

struct pressio;
struct pressio_compressor;
struct pressio_metrics;

/**
 * gets a reference to a new instance of libpressio; initializes the library if necessary
 * \returns a pointer to a library instance
 */
struct pressio* pressio_instance();


/**
 * \param[in] library the pointer to the library
 * \returns informs the library that this instance is no longer required; the pointer passed becomes invalid
 */
void pressio_release(struct pressio* library);

/**
 * \param[in] library the pointer to the library
 * \param[in] compressor_id the compressor to use
 * \returns non-owning pointer to the requested instantiated pressio compressor; it may return the same pointer on multiple calls
 * \see pressio_features for a list of available compressors
 */
struct pressio_compressor* pressio_get_compressor(struct pressio* library, const char* compressor_id);

/**
 * creates a possibly composite metrics structure
 *
 * \param[in] library the pointer to the library
 * \param[in] metrics a list of c-strings containing the list of metrics requested
 * \param[in] num_metrics the number of metrics requested
 * \returns a new pressio_metrics structure
 */
struct pressio_metrics* pressio_new_metrics(struct pressio* library, const char* metrics[], int num_metrics);

/**
 * \param[in] library the pointer to the library
 * \returns a machine-readable error code for the last error on the library object
 */
int pressio_error_code(struct pressio* library);

/**
 * \param[in] library the pointer to the library
 * \returns a human-readable error message for the last error on the library object
 */
const char* pressio_error_msg(struct pressio* library);

/**
 * it will not return more information than the tailored functions below
 * \returns a string with version and feature information
 */
const char* pressio_version();
/**
 * \returns a string containing all the features supported by this version separated by a space.  Some features are compressors, but not all are.
 * \see pressio_get_compressor the compressor_ids may be passed to pressio_get_compressor
 */
const char* pressio_features();
/**
 * \returns a string containing all the compressors supported by this version separated by a space
 * \see pressio_get_supported_compressors the compressor_ids may be passed to pressio_get_compressor
 */
const char* pressio_supported_compressors();
/**
 * \returns the major version of the library
 */
unsigned int pressio_major_version();
/**
 * \returns the minor version of the library
 */
unsigned int pressio_minor_version();
/**
 * \returns the patch version of the library
 */
unsigned int pressio_patch_version();
#endif

#ifdef __cplusplus 
}
#endif 
