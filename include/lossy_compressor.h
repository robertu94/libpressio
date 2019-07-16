#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIBLOSSY_COMPRESSOR_H
#define LIBLOSSY_COMPRESSOR_H

/*! \file
 *  \brief Compress, decompress, and configure lossy and lossless compressors
 * */

struct lossy_compressor;
struct lossy_data;
struct lossy_options;

//option getting/setting functions
/*!
 * \returns a lossy options struct that represents the current options of the compressor
 * \param[in] compressor which compressor to get options for
 * It may return a pointer to the same memory on multiple calls.
 */
struct lossy_options* lossy_compressor_get_options(struct lossy_compressor const* compressor);
/*!
 * sets the options for the lossy_compressor
 * \param[in] compressor which compressor to get options for
 * \param[in] options the options to set
 * \returns 0 if successful, or non-zero code on error.
 */
int lossy_compressor_set_options(struct lossy_compressor* compressor, struct lossy_options const * options);

//compression/decompression functions
/*!
 * compresses the data in data using the specified compressor
 * \param[in] compressor compressor to be used
 * \param[in] input data to be compressed and associated metadata
 * \param[in,out] output output data and metadata that may be used for how to output information provide this information if possible
 * \returns returns 0 if there is no error or a compressor specific code there is an error
 */
int lossy_compressor_compress(struct lossy_compressor* compressor, struct lossy_data * input, struct lossy_data** output);
/*!
 * decompresses the compressed data using the specified compressor
 * calling this without calling liblossy_compressor_set_options() has undefined behavior
 * decompressing with a compressor with different settings than used for compression has undefined behavior
 *
 * \param[in] compressor compressor to be used
 * \param[in] input data to be decompressed and associated metadata
 * \param[in,out] output output data and metadata that may be used for how to output information provide this information if possible
 * \returns returns 0 if there is no error or a compressor specific code there is an error
 */
int lossy_compressor_decompress(struct lossy_compressor* compressor, struct lossy_data * data, struct lossy_data** output);

/**
 * \param[in] compressor the compressor to query
 * \returns last error code for the compressor
 */
int lossy_compressor_error_code(struct lossy_compressor const* compressor);

/**
 * \param[in] compressor the compressor to query
 * \returns last error message for the compressor
 */
const char* lossy_compressor_error_msg(struct lossy_compressor const* compressor);


/*!
 * Get the version and feature information.  The version string may include more information than major/minor/patch provide.
 * \param[in] compressor the compressor to query
 * \returns a implementation specific version string
 */
const char* lossy_compressor_version(struct lossy_compressor const* compressor);
/*!
 * \param[in] compressor the compressor to query
 * \returns the major version number or 0 if there is none
 */
int lossy_compressor_major_version(struct lossy_compressor const* compressor);
/*!
 * \param[in] compressor the compressor to query
 * \returns the major version number or 0 if there is none
 */
int lossy_compressor_minor_version(struct lossy_compressor const* compressor);
/*!
 * \param[in] compressor the compressor to query
 * \returns the major version number or 0 if there is none
 */
int lossy_compressor_patch_version(struct lossy_compressor const* compressor);




#endif

#ifdef __cplusplus
}
#endif
