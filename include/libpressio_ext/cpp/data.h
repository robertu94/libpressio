#ifndef PRESSIO_DATA_CPP_H
#define PRESSIO_DATA_CPP_H



#include <vector>
#include <cstdlib>
#include <cstring>
#include <utility>
#include "pressio_data.h"
#include "pressio_dtype.h"

/**
 * \file
 * \brief C++ pressio_data interface
 */

namespace {
  template <class T>
  size_t data_size_in_elements(size_t dimensions, T const dims[]) {
    size_t totalsize = 1;
    for (size_t i = 0; i < dimensions; ++i) {
      totalsize *= dims[i];
    }
    return totalsize;
  }
  template <class T>
  size_t data_size_in_bytes(pressio_dtype type, T const dimensions, T const dims[]) {
    return data_size_in_elements(dimensions, dims) * pressio_dtype_size(type);
  }
}

/**
 * represents a data buffer that may or may not be owned by the class
 */
struct pressio_data {

  /**  
   * allocates a new empty data buffer
   *
   * \param[in] dtype the type the buffer will contain
   * \param[in] dimensions the dimensions of the expected buffer
   * \returns an empty data object (i.e. has no data)
   * \see pressio_data_new_empty
   * */
  static pressio_data empty(const pressio_dtype dtype, std::vector<size_t> const& dimensions) {
    return pressio_data(dtype, nullptr, nullptr, nullptr, dimensions.size(), dimensions.data());
  }
  /**  
   * creates a non-owning reference to data
   *
   * \param[in] dtype the type of the buffer
   * \param[in] data the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \returns an non-owning data object (i.e. calling pressio_data_free will not deallocate this memory)
   * \see pressio_data_new_nonowning
   * */
  static pressio_data nonowning(const pressio_dtype dtype, void* data, std::vector<size_t> const& dimensions) {
    return pressio_data::nonowning(dtype, data, dimensions.size(), dimensions.data());
  }
  /**  
   * creates a copy of a data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] src the buffer to copy \param[in] dimensions the dimensions of the buffer \returns an owning copy of the data object \see pressio_data_new_copy */
  static pressio_data copy(const enum pressio_dtype dtype, void* src, std::vector<size_t> const& dimensions) {
    return pressio_data::copy(dtype, src, dimensions.size(), dimensions.data());
  }
  /**  
   * creates a copy of a data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_owning
   * */
  static pressio_data owning(const pressio_dtype dtype, std::vector<size_t> const& dimensions) {
    return pressio_data::owning(dtype, dimensions.size(), dimensions.data());
  }
  /**  
   * takes ownership of an existing data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] data the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \param[in] deleter the method to call to free the buffer or null to not free the data
   * \param[in] metadata the metadata passed to the deleter function
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_move
   * */
  static pressio_data move(const pressio_dtype dtype,
      void* data,
      std::vector<size_t> const& dimensions,
      pressio_data_delete_fn deleter,
      void* metadata) {
    return pressio_data::move(dtype, data, dimensions.size(), dimensions.data(), deleter, metadata);
  }

  /**  
   * allocates a new empty data buffer
   *
   * \param[in] dtype the type the buffer will contain
   * \param[in] num_dimensions the length of dimensions
   * \param[in] dimensions the dimensions of the expected buffer
   * \returns an empty data object (i.e. has no data)
   * \see pressio_data_new_empty
   * */
  static pressio_data empty(const pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]) {
    return pressio_data(dtype, nullptr, nullptr, nullptr, num_dimensions, dimensions);
  }

  /**  
   * creates a non-owning reference to data
   *
   * \param[in] dtype the type of the buffer
   * \param[in] data the buffer
   * \param[in] num_dimensions the number of entries in dimensions
   * \param[in] dimensions the dimensions of the data
   * \returns an non-owning data object (i.e. calling pressio_data_free will not deallocate this memory)
   * \see pressio_data_new_nonowning
   * */
  static pressio_data nonowning(const pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[]) {
    return pressio_data(dtype, data, nullptr, nullptr, num_dimensions, dimensions);
  }

  /**  
   * creates a copy of a data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] src the buffer to copy
   * \param[in] num_dimensions the number of entries in dimensions
   * \param[in] dimensions the dimensions of the data
   * \returns an owning copy of the data object
   * \see pressio_data_new_copy
   * */
  static pressio_data copy(const enum pressio_dtype dtype, void* src, size_t const num_dimensions, size_t const dimensions[]) {
    size_t bytes = data_size_in_bytes(dtype, num_dimensions, dimensions);
    void* data = malloc(bytes);
    memcpy(data, src, bytes);
    return pressio_data(dtype, data, nullptr, pressio_data_libc_free_fn, num_dimensions, dimensions);
  }

  /**  
   * creates a copy of a data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] num_dimensions the number of entries in dimensions
   * \param[in] dimensions the dimensions of the data
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_owning
   * */
  static pressio_data owning(const pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]) {
    void* data = malloc(data_size_in_bytes(dtype, num_dimensions, dimensions));
    return pressio_data(dtype, data, nullptr, pressio_data_libc_free_fn, num_dimensions, dimensions);
  }


  /**  
   * takes ownership of an existing data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] data the buffer
   * \param[in] num_dimensions the number of entries in dimensions
   * \param[in] dimensions the dimensions of the data
   * \param[in] deleter the method to call to free the buffer or null to not free the data
   * \param[in] metadata the metadata passed to the deleter function
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_move
   * */
  static pressio_data move(const pressio_dtype dtype,
      void* data,
      size_t const num_dimensions,
      size_t const dimensions[],
      pressio_data_delete_fn deleter,
      void* metadata) {
    return pressio_data(dtype, data, metadata, deleter, num_dimensions, dimensions);
  }



  ~pressio_data() {
    if(deleter!=nullptr) deleter(data_ptr,metadata_ptr);
  };

  //disable copy constructors/assignment to handle case of non-owning pointers
  /** copy-assignment is disabled */
  pressio_data& operator=(pressio_data const& rhs)=delete;
  /** copy-constructing is disabled */
  pressio_data(pressio_data const& rhs)=delete; 

  
  /**
   * move-constructor
   *
   * \param[in] rhs the data buffer to move from
   * \returns a reference to the object moved into
   */
  pressio_data(pressio_data&& rhs) noexcept:
    data_dtype(rhs.data_dtype),
    data_ptr(std::exchange(rhs.data_ptr, nullptr)),
    metadata_ptr(std::exchange(rhs.metadata_ptr, nullptr)),
    deleter(std::exchange(rhs.deleter, nullptr)),
    dims(std::exchange(rhs.dims, {})) {}
  
  /**
   * move-assignment operator
   *
   * \param[in] rhs the data buffer to move from
   * \returns a l-value reference to the object moved into
   */
  pressio_data& operator=(pressio_data && rhs) noexcept {
    data_dtype = rhs.data_dtype,
    data_ptr = std::exchange(rhs.data_ptr, nullptr),
    metadata_ptr = std::exchange(rhs.metadata_ptr, nullptr),
    deleter = std::exchange(rhs.deleter, nullptr),
    dims = std::exchange(rhs.dims, {});
    return *this;
  }

  /**
   * \returns a non-owning pointer to the data
   */
  void* data() const {
    return data_ptr;
  }
  
  /**
   * \returns the data type of the buffer
   */
  pressio_dtype dtype() const {
    return data_dtype;
  }

  /**
   * \returns the number of dimensions
   */
  size_t num_dimensions() const {
    return dims.size();
  }

  /**
   * \returns a copy of the vector of dimensions
   */
  std::vector<size_t> dimensions() const {
    return dims;
  }

  /**
   * \param idx the specific index to query
   * \returns a specific dimension of the buffer of zero if the index exceeds dimensions()
   */
  size_t get_dimension(size_t idx) const {
    if(idx >= num_dimensions()) return 0;
    else return dims[idx];
  }

  /**
   * \returns the size of the buffer in bytes
   */
  size_t size_in_bytes() const {
    return data_size_in_bytes(data_dtype, num_dimensions(), dims.data());
  }

  /**
   * \returns the size of the buffer in elements
   */
  size_t num_elements() const {
    return data_size_in_elements(num_dimensions(), dims.data());
  }

  /**
   * Copies a set of blocks of data from the data stream to a new pressio_data structure
   *
   * \param[in] start the position in the array to start iterating, start[i]>=0
   * \param[in] stride the number of blocks to skip in each direction, stride[i] >=1
   * \param[in] count the number of blocks to copy, count[i] >= 1
   * \param[in] block the dimensions of the block to copy, block[i] >= 1
   *
   * \returns the copied blocks, or an empty structure if an error occurs
   */
  pressio_data select(std::vector<size_t> const& start = {},
      std::vector<size_t> const& stride = {},
      std::vector<size_t> const& count = {},
      std::vector<size_t> const& block = {}) const;
  

  private:
  /**
   * constructor use the static methods instead
   * \param dtype the type of the data
   * \param data the buffer to use
   * \param metadata the meta data to pass to the deleter function
   * \param num_dimensions the number of dimensions to represent
   * \param dimensions of the data
   */
  pressio_data(const pressio_dtype dtype,
      void* data,
      void* metadata,
      void (*deleter)(void*, void*),
      size_t const num_dimensions,
      size_t const dimensions[]):
    data_dtype(dtype),
    data_ptr(data),
    metadata_ptr(metadata),
    deleter(deleter),
    dims(dimensions, dimensions+num_dimensions)
  {}
  pressio_dtype data_dtype;
  void* data_ptr;
  void* metadata_ptr;
  void (*deleter)(void*, void*);
  std::vector<size_t> dims;
};

#endif /* end of include guard: PRESSIO_DATA_CPP_H */
