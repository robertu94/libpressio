#include "stddef.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIBPRESSIO_COMPRESSOR_H
#define LIBPRESSIO_COMPRESSOR_H

/*! \file
 *  \brief Compress, decompress, and configure pressio and lossless compressors
 * */

struct pressio_compressor;
struct pressio_data;
struct pressio_options;

/*!
 * \param[in] compressor deallocates a reference to a compressor.
 */
void pressio_compressor_release(struct pressio_compressor* compressor);

//option getting/setting functions
/*!
 * \returns a pressio options struct that represents the documentation for the compressor
 * \param[in] compressor which compressor to get documentation for
 */
struct pressio_options* pressio_compressor_get_documentation(struct pressio_compressor const* compressor);

//option getting/setting functions
/*!
 * \returns a pressio options struct that represents the compile time configuration of the compressor
 * \param[in] compressor which compressor to get compile-time configuration for
 */
struct pressio_options* pressio_compressor_get_configuration(struct pressio_compressor const* compressor);

/*!
 * \returns a pressio options struct that represents the current options of the compressor
 * \param[in] compressor which compressor to get options for
 */
struct pressio_options* pressio_compressor_get_options(struct pressio_compressor const* compressor);
/*!
 * sets the options for the pressio_compressor.  Compressors MAY choose to ignore
 * some subset of options passed in if there valid but conflicting settings
 * (i.e. two settings that adjust the same underlying configuration).
 * Additionally, if one of the two settings is a generic one (i.e. pressio:abs)
 * the compressor specific version should prevail.
 * Compressors SHOULD return an error value if configuration
 * failed due to a missing required setting or an invalid one. Users MAY
 * call pressio_compressor_error_msg() to get more information about the warnings or errors
 * Compressors MUST ignore any and all options set that they do not support.
 *
 * \param[in] compressor which compressor to get options for
 * \param[in] options the options to set
 * \returns 0 if successful, positive values on errors, negative values on warnings
 *
 * \see pressio_compressor_error_msg
 */
int pressio_compressor_set_options(struct pressio_compressor* compressor, struct pressio_options const * options);
/*!
 * Validates that only defined options have been set.  This can be useful for programmer errors.
 * This function should NOT be used with any option structure which contains options for multiple compressors.
 * Other checks MAY be preformed implementing compressors.
 * 
 * \param[in] compressor which compressor to validate the options struct against
 * \param[in] options which options set to check against.  It should ONLY contain options returned by pressio_compressor_get_options
 * \returns 0 if successful, 1 if there is an error.  On error, an error message is set in pressio_compressor_error_msg.
 * \see pressio_compressor_error_msg to get the error message
 */
int pressio_compressor_check_options(struct pressio_compressor* compressor, struct pressio_options const * options);


//compression/decompression functions
/*!
 * compresses the data in data using the specified compressor
 * \param[in] compressor compressor to be used
 * \param[in] input data to be compressed and associated metadata
 * \param[in,out] output 
 *    when passed in, \c output MAY contain metadata (type, dimentions) and additionally MAY contain a buffer.
 *    pressio_data_free will be called on the pointer passed into this function if a new owning pressio_data structure is returned.
 *    when passed out, \c output will contain either:
 *      1. The same pointer passed in to \c output if the compressor supports
 *         using a provided buffer for the results of the compression and contains has a buffer.  It is
 *         recommended the user provide an owning pressio_data structure if passing a pressio_data structure with a buffer.
 *      2. A new owning pressio_data structure with the results of the compression
 * \returns 0 if successful, positive values on errors, negative values on warnings
 */
int pressio_compressor_compress(struct pressio_compressor* compressor, const struct pressio_data *input, struct pressio_data* output);
/*!
 * decompresses the compressed data using the specified compressor
 * calling this without calling libpressio_compressor_set_options() has undefined behavior
 * decompressing with a compressor with different settings than used for compression has undefined behavior
 *
 * \param[in] compressor compressor to be used
 * \param[in] input data to be decompressed and associated metadata
 * \param[in,out] output 
 *    when passed in, \c output SHOULD contain the metadata (type, dimentions) for the output of the compression if available.
 *    pressio_data_free will be called the pointer passed in during this function.
 *    when passed out, it will contain either
 *      1. The same pointer passed in to \c output if the compressor supports
 *         using a provided buffer for the results of the compression and contains has a buffer.  It is
 *         recommended the user provide an owning pressio_data structure if passing a pressio_data structure with a buffer.
 *      2. A new owning pressio_data structure with the results of the compression
 * \returns 0 if successful, positive values on errors, negative values on warnings
 * \see pressio_data_new_empty often used as the pointer passed into \c output
 * \see pressio_data_new_move often used as the pointer passed out of \c output
 */
int pressio_compressor_decompress(struct pressio_compressor* compressor, const struct pressio_data *input, struct pressio_data* output);

/**
 * \param[in] compressor the compressor to get results from
 * \returns a pressio_options structure containing the metrics returned by the provided metrics plugin
 * \see libpressio_metricsplugin for how to compute results
 */
struct pressio_options* pressio_compressor_get_metrics_results(struct pressio_compressor const* compressor);

/**
 * \param[in] compressor the compressor to get the metrics plugin for
 * \returns the current pressio_metrics* structure
 */
struct pressio_metrics* pressio_compressor_get_metrics(struct pressio_compressor const* compressor);

/**
 * \param[in] compressor the compressor to set metrics plugin for
 * \param[in] plugin the configured libpressio_metricsplugin plugin to use
 */
void pressio_compressor_set_metrics(struct pressio_compressor* compressor, struct pressio_metrics* plugin);


/**
 * Gets the options for a metrics structure
 * \param[in] compressor the metrics structure to get options for
 * \returns a new pressio_options structure with the options for the metrics
 */
struct pressio_options* pressio_compressor_metrics_get_options(struct pressio_compressor const* compressor);

/**
 * Gets the options for a metrics structure
 * \param[in] compressor the compressor structure to get metrics options for
 * \param[in] options the options to set
 * \returns 0 if successful, positive values on errors, negative values on warnings
 */
int pressio_compressor_metrics_set_options(struct pressio_compressor const* compressor, struct pressio_options const* options);

/**
 * \param[in] compressor the compressor to query
 * \returns last error code for the compressor
 */
int pressio_compressor_error_code(struct pressio_compressor const* compressor);

/**
 * \param[in] compressor the compressor to query
 * \returns last error message for the compressor
 */
const char* pressio_compressor_error_msg(struct pressio_compressor const* compressor);


/*!
 * Get the version and feature information.  The version string may include more information than major/minor/patch provide.
 * \param[in] compressor the compressor to query
 * \returns a implementation specific version string
 */
const char* pressio_compressor_version(struct pressio_compressor const* compressor);
/*!
 * \param[in] compressor the compressor to query
 * \returns the major version number or 0 if there is none
 */
int pressio_compressor_major_version(struct pressio_compressor const* compressor);
/*!
 * \param[in] compressor the compressor to query
 * \returns the major version number or 0 if there is none
 */
int pressio_compressor_minor_version(struct pressio_compressor const* compressor);
/*!
 * \param[in] compressor the compressor to query
 * \returns the major version number or 0 if there is none
 */
int pressio_compressor_patch_version(struct pressio_compressor const* compressor);

/**
 * Clones a compressor and its configuration including metrics information
 *
 * \param[in] compressor the compressor to clone. It will not be modified
 *                       except to modify a reference count as needed.
 * \returns a pointer to a new compressor plugin reference which is equivalent
 *          to the compressor cloned.  It the compressor is not thread safe, it may
 *          return a new reference to the same object.
 *                
 */
struct pressio_compressor* pressio_compressor_clone(struct pressio_compressor* compressor);


/**
 * Returns the name this compressor uses its keys
 *
 * \param[in] compressor the compressor to get the prefix for
 * \returns the prefix for this compressor
 */
const char* pressio_compressor_get_prefix(const struct pressio_compressor* compressor);

/**
 * Assign a new name to a compressor.  Names are used to prefix options in meta-compressors.
 *
 * sub-compressors will be renamed either by the of the sub-compressors prefix
 * or by the $prefix:name configuration option
 *
 * i.e. for some new_name and a compressor with prefix foo and subcompressors
 * with prefixs "abc", "def", "ghi" respectively
 *
 * - if foo:names = ['one', 'two', 'three'], then the names will be `$new_name/one, $new_name/two $new_name/three
 * - otherwise the names will be $new_name/abc, $new_name/def, $new_name/ghi
 *
 * \param[in] compressor the compressor to get the name of
 * \param[in] new_name the name to set
 */
void pressio_compressor_set_name(struct pressio_compressor* compressor, const char* new_name);

/**
 * Get the name of a compressor
 * \param[in] compressor the compressor to get the name of
 * \returns a string with the compressor name. The string becomes invalid if
 *          the name is set_name is called.
 */
const char* pressio_compressor_get_name(struct pressio_compressor const* compressor);

/**
 * reports the level of thread safety supported by the compressor.
 *
 * Compressors MUST report a thread safety by setting the pressio:thread_safe
 * option on the object returned by get_configuration and the level supported
 * by the plug-in.
 *
 * Safety is defined in terms of if the both of the following sequence of calls
 * from multiple threads can be made without a data race:
 *
 * \code{.c}
 * pressio_compressor_get_options(...)
 * pressio_compressor_set_options(...)
 * \endcode
 *
 * and
 *
 * \code{.c}
 * pressio_compressor_get_options(...)
 * pressio_compressor_set_options(...)
 * \endcode
 *
 * and
 *
 * \code{.c}
 * pressio_compressor_set_options(...)
 * pressio_compressor_compress(...)
 * pressio_compressor_error_code(...)
 * pressio_compressor_error_msg(...)
 * \endcode
 *
 * and
 *
 * \code{.c}
 * pressio_compressor_set_options(...)
 * pressio_compressor_decompress(...)
 * pressio_compressor_error_code(...)
 * pressio_compressor_error_msg(...)
 * \endcode
 *
 * All metrics plugins MUST support pressio_thread_safety_multiple (i.e. safe as long as different objects are used)
 *
 */
enum pressio_thread_safety {
  /** use of this compressor in a multi-threaded environment is unsafe. */
  pressio_thread_safety_single = 0,
  /** calls are safe if and only if only one thread will execute the above sequences of calls to any instance of the compressor at a time*/
  pressio_thread_safety_serialized = 1,
  /** calls are safe if and only if only one thread will execute the above sequences of calls to an instance of the compressor at a time*/
  pressio_thread_safety_multiple = 2,
};


/**
 * compress multiple data buffers in one api call. Underlying implementations may do this in serial or parallel
 *
 * \param[in] compressor the compressor object that will perform compression
 * \param[in] in array of input buffers
 * \param[in] num_inputs the number of elements of the "in" array
 * \param[in,out] out array of compressed data buffers
 *  When passed in, each element of out MAY contain metadata (type, dimentions), and may additionally contain a buffer.
 *    pressio_data_free will need to be called on the object returned from this buffer.
 *  When passed out, each elements of out will contain either
 *      1. The same pointer passed in to \c output if the compressor supports
 *         using a provided buffer for the results of the compression and contains has a buffer.  It is
 *         recommended the user provide an owning pressio_data structure if passing a pressio_data structure with a buffer.
 *      2. A new owning pressio_data structure with the results of the compression
 * \param[in] num_outputs the number of elements of the "out" array
 * \returns 0 if successful, 1 if there is an error.  On error, an error message is set in pressio_compressor_error_msg.
 *
 */
int pressio_compressor_compress_many(struct pressio_compressor* compressor,
    struct pressio_data const*const in[], size_t num_inputs,
    struct pressio_data * out[], size_t num_outputs
    );

/**
 * decompress multiple data buffers in one api call.  Underlying implementations may do this in serial or parallel
 *
 * \param[in] compressor the compressor object that will perform compression
 * \param[in] in array of input buffers
 * \param[in] num_inputs the number of elements of the "in" array
 * \param[in,out] out array of compressed data buffers
 *  When passed in, each element of out MAY contain metadata (type, dimentions), and may additionally contain a buffer.
 *    pressio_data_free will need to be called on the object returned from this buffer.
 *  When passed out, each elements of out will contain either
 *      1. The same pointer passed in to \c output if the compressor supports
 *         using a provided buffer for the results of the compression and contains has a buffer.  It is
 *         recommended the user provide an owning pressio_data structure if passing a pressio_data structure with a buffer.
 *      2. A new owning pressio_data structure with the results of the compression
 * \param[in] num_outputs the number of elements of the "out" array
 * \returns 0 if successful, 1 if there is an error.  On error, an error message is set in pressio_compressor_error_msg.
 */
int pressio_compressor_decompress_many(struct pressio_compressor* compressor,
    struct pressio_data const*const in[], size_t num_inputs,
    struct pressio_data * out[], size_t num_outputs
    );

#endif

#ifdef __cplusplus
}
#endif
