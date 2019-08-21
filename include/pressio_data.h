#include <stddef.h>
#include "pressio_dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIBPRESSIO_DATA_H
#define LIBPRESSIO_DATA_H

/*! \file
 *  \brief an abstraction for a contagious memory region of a specified type
 */

struct pressio_data;

/**
 * signature for a custom deleter for pressio_data
 *
 * \param[in] data to be deallocated
 * \param[in] metadata  metadata passed to pressio_data_new_move, if allocated, it too should be freed
 *
 * \see pressio_data_new_move
 */
typedef void (*pressio_data_delete_fn)(void* data, void* metadata);

/**
 * a custom deleter that uses libc's free and ignores the metadata
 * \param[in] data to be deallocated with free()
 * \param[in] metadata  ignored
 */
void pressio_data_libc_free_fn (void* data, void* metadata);

/** 
 *  allocates a new pressio_data structure, it does NOT take ownership of data.
 *
 *  \param[in] dtype type of the data stored by the pointer
 *  \param[in] data the actual data to be represented
 *  \param[in] num_dimensions the number of dimensions; must match the length of dimensions
 *  \param[in] dimensions an array corresponding to the dimensions of the data, a copy is made of this on construction
 */
struct pressio_data* pressio_data_new_nonowning(const enum pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[]);
/** 
 *  allocates a new pressio_data structure and corresponding data and copies data in from the specified source
 *  use this function when the underlying data pointer may be deleted
 *
 *  \param[in] dtype type of the data stored by the pointer
 *  \param[in] src the data to be copied into the data structure
 *  \param[in] num_dimensions the number of dimensions; must match the length of dimensions
 *  \param[in] dimensions an array corresponding to the dimensions of the data, a copy is made of this on construction
 */
struct pressio_data* pressio_data_new_copy(const enum pressio_dtype dtype, void* src, size_t const num_dimensions, size_t const dimensions[]);
/** 
 *  allocates a new pressio_data structure and corresponding data. The corresponding data is uninitialized
 *
 *  \param[in] dtype type of the data stored by the pointer
 *  \param[in] num_dimensions the number of dimensions; must match the length of dimensions
 *  \param[in] dimensions an array corresponding to the dimensions of the data, a copy is made of this on construction
 *
 *  \see pressio_data_ptr to access the allocated data
 */
struct pressio_data* pressio_data_new_owning(const enum pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]);
/** 
 *  allocates a new pressio_data without data.
 *
 *  \param[in] dtype type of the data stored by the pointer
 *  \param[in] num_dimensions the number of dimensions; must match the length of dimensions
 *  \param[in] dimensions an array corresponding to the dimensions of the data, a copy is made of this on construction
 *  \see pressio_compressor_compress to provide output buffer meta data.
 *  \see pressio_compressor_decompress to provide output buffer meta data.
 */
struct pressio_data* pressio_data_new_empty(const enum pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]);

/**
 * allocates a new pressio_data structure using data that was already allocated.
 * data referenced by this structure will be deallocated using the function
 * deleter when the pressio_data structure is freed.
 *
 * \param[in] dtype the type of the data to store
 * \param[in] data the pointer to be "moved" into the pressio_data structure
 * \param[in] num_dimensions the number of dimensions; must match the length of dimensions
 * \param[in] dimensions an array corresponding to the dimensions of the data, a copy is made of this on construction
 * \param[in] deleter the function to be called when pressio_data_free is called on this structure
 * \param[in] metadata passed to the deleter function when pressio_data_free is called, it may be null if unneeded
 */
struct pressio_data* pressio_data_new_move(const enum pressio_dtype dtype, void* data, size_t const num_dimensions,
    size_t const dimensions[], pressio_data_delete_fn deleter, void* metadata);

/**
 * frees a pressio_data structure, but not the underlying data
 * \param[in] data data structure to free
 */
void pressio_data_free(struct pressio_data* data);

/**
 * allocates a buffer and then copies the underlying memory to the new buffer
 * \param[in] data the pressio data to copy from
 * \param[out] out_bytes the number of bytes that were copied
 */
void* pressio_data_copy(struct pressio_data const* data, size_t* out_bytes);

/**
 * non-owning access to the first element of the raw data
 * \param[in] data the pressio data to query
 * \param[out] out_bytes the number of bytes that follow this pointer (ignored if NULL is passed)
 * \returns a non-owning type-pruned pointer to the first element of the raw data
 * \see pressio_data_dtype to get the data-type 
 */
void* pressio_data_ptr(struct pressio_data const* data, size_t* out_bytes);
/**
 * \param[in] data the pressio data to query
 * \returns an integer code corresponding to the data-type
 */
enum pressio_dtype pressio_data_dtype(struct pressio_data const* data);
/**
 * \param[in] data the pressio data to query
 * \returns an integer code corresponding to the data-type
 */
bool pressio_data_has_data(struct pressio_data const* data);
/**
 * \param[in] data the pressio data to query
 * \returns the number of dimensions contained in the object
 */
size_t pressio_data_num_dimensions(struct pressio_data const* data);
/**
 * returns the value of a given dimension. 
 * \param[in] data the pressio data to query
 * \param[in] dimension zero indexed dimension
 * \returns the dimension or 0 If the dimension requested exceeds num_dimensions
 */
size_t pressio_data_get_dimension(struct pressio_data const* data, size_t const dimension);

/**
 * returns the number of bytes to represent the data
 * \param[in] data the pressio data to query
 * \returns the number of bytes to represent the data
 */
size_t pressio_data_get_bytes(struct pressio_data const* data);

/**
 * returns the total number of elements to represent the data
 * \param[in] data the pressio data to query
 * \returns the total number of elements to represent the data
 */
size_t pressio_data_num_elements(struct pressio_data const* data);

#endif

#ifdef __cplusplus
}
#endif
