#ifdef __cplusplus 
extern "C" {
#endif 

#ifndef LIBLOSSY_H
#define LIBLOSSY_H

/*! \file
 * \brief Lossy compressor loader
 */

struct lossy;
struct lossy_compressor;

/**
 * gets a reference to a possibly shared instance of liblossy; initializes the library if necessary
 * \returns a pointer to a library instance
 */
struct lossy* lossy_instance();


/**
 * \param[in,out] library the pointer to the library
 * \returns informs the library that this instance is no longer required; the pointer passed becomes invalid
 */
void lossy_release(struct lossy** library);

/**
 * \param[in] library the pointer to the library
 * \param[in] compressor_id the compressor to use
 * \returns non-owning pointer to the requested instantiated lossy compressor; it may return the same pointer on multiple calls
 * \see lossy_features for a list of available compressors
 */
struct lossy_compressor* lossy_get_compressor(struct lossy* library, const char* const compressor_id);

/**
 * \param[in] library the pointer to the library
 * \returns a machine-readable error code for the last error on the library object
 */
int lossy_error_code(struct lossy* library);

/**
 * \param[in] library the pointer to the library
 * \returns a human-readable error message for the last error on the library object
 */
const char* lossy_error_msg(struct lossy* library);

/**
 * \returns a string with version and feature information
 */
const char* lossy_version();
/**
 * \returns a string containing all the compressor_ids supported by this version separated by a space
 * it will not return more information than the tailored functions below
 * \see lossy_get_compressor the compressor_ids may be passed to lossy_get_compressor
 */
const char* lossy_features();
/**
 * \returns the major version of the library
 */
unsigned int lossy_major_version();
/**
 * \returns the minor version of the library
 */
unsigned int lossy_minor_version();
/**
 * \returns the patch version of the library
 */
unsigned int lossy_patch_version();
#endif

#ifdef __cplusplus 
}
#endif 
