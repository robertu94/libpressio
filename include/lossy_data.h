#include <stddef.h>
#include "lossy_dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIBLOSSY_DATA_H
#define LIBLOSSY_DATA_H

/*! \file
 *  \brief an abstraction for a contagious memory region of a specified type
 */

struct lossy_data;

/** 
 *  allocates a new lossy_data structure, it does NOT take ownership of data.
 *  \param[in] dtype type of the data stored by the pointer
 *  \param[in] data the actual data to be represented
 *  \param[in] num_dimentions the number of dimentions; must match the length of dimensions
 *  \param[in] dimensions an array corresponding to the dimentions of the data, a copy is made of this on construction
 */
struct lossy_data* lossy_data_new(const enum lossy_dtype dtype, void* data, size_t const num_dimentions, size_t const dimensions[]);
/** 
 *  allocates a new lossy_data without data.  
 *  \param[in] dtype type of the data stored by the pointer
 *  \param[in] num_dimentions the number of dimentions; must match the length of dimensions
 *  \param[in] dimensions an array corresponding to the dimensions of the data, a copy is made of this on construction
 *  \see lossy_compressor_compress to provide output buffer meta data.
 *  \see lossy_compressor_decompress to provide output buffer meta data.
 */
struct lossy_data* lossy_data_new_empty(const enum lossy_dtype dtype, size_t const num_dimentions, size_t const dimensions[]);

/**
 * frees a lossy_data structure, but not the underlying data
 * \param[in] data data structure to free
 */
void lossy_data_free(struct lossy_data* data);

/**
 * allocates a buffer and then copies the underlying memory to the new buffer
 * \param[in] data the lossy data to copy from
 * \param[out] out_bytes the number of bytes that were copied
 */
void* lossy_data_copy(struct lossy_data const* data, size_t* out_bytes);

/**
 * non-owning access to the first element of the raw data
 * \param[in] data the lossy data to query
 * \param[out] out_bytes the number of bytes that follow this pointer (ignored if NULL is passed)
 * \returns a non-owning type-pruned pointer to the first element of the raw data
 * \see lossy_data_dtype to get the data-type 
 */
void* lossy_data_ptr(struct lossy_data const* data, size_t* out_bytes);
/**
 * \param[in] data the lossy data to query
 * \returns an integer code corresponding to the data-type
 */
enum lossy_dtype lossy_data_dtype(struct lossy_data const* data);
/**
 * \param[in] data the lossy data to query
 * \returns an integer code corresponding to the data-type
 */
bool lossy_data_has_data(struct lossy_data const* data);
/**
 * \param[in] data the lossy data to query
 * \returns the number of dimensions contained in the object
 */
size_t lossy_data_num_dimentions(struct lossy_data const* data);
/**
 * returns the value of a given dimension. 
 * \param[in] data the lossy data to query
 * \param[in] dimension zero indexed dimension
 * \returns the dimension or 0 If the dimension requested exceeds num_dimentions
 */
size_t lossy_data_get_dimention(struct lossy_data const* data, size_t const dimention);

#endif

#ifdef __cplusplus
}
#endif
