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

//option getting/setting functions
/*!
 * \returns a pressio options struct that represents the current options of the compressor
 * \param[in] compressor which compressor to get options for
 * It may return a pointer to the same memory on multiple calls.
 */
struct pressio_options* pressio_compressor_get_options(struct pressio_compressor const* compressor);
/*!
 * sets the options for the pressio_compressor.  Compressors MAY choose to ignore
 * some subset of options passed in if there valid but conflicting settings
 * (i.e. two settings that adjust the same underlying configuration).
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
int pressio_compressor_compress(struct pressio_compressor* compressor, struct pressio_data * input, struct pressio_data** output);
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
 *    when passed out, it will contain an owning pressio_data structure with the result of the decompression.
 * \returns 0 if successful, positive values on errors, negative values on warnings
 * \see pressio_data_new_empty often used as the pointer passed into \c output
 * \see pressio_data_new_move often used as the pointer passed out of \c output
 */
int pressio_compressor_decompress(struct pressio_compressor* compressor, struct pressio_data * input, struct pressio_data** output);

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




#endif

#ifdef __cplusplus
}
#endif
