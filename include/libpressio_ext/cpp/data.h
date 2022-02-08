#ifndef PRESSIO_DATA_CPP_H
#define PRESSIO_DATA_CPP_H



#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <algorithm>
#include "pressio_data.h"
#include "libpressio_ext/cpp/dtype.h"
#include "std_compat/utility.h"

/**
 * \file
 * \brief C++ pressio_data interface
 */

/**
 * pressio_data_delete_fn for handling deleting data allocated with new T[];
 */
template <class T>
pressio_data_delete_fn pressio_new_free_fn() {
  return [](void* data, void*){
  T* data_t = static_cast<T*>(data);
  delete[] data_t;
  };
}


/**
 * \param[in] dimensions the number of dimensions of the data object
 * \param[in] dims the actual of dimensions of the data object
 * \returns the size of a data object in elements
 */
size_t data_size_in_elements(size_t dimensions, size_t const dims[]);
/**
 * \param[in] type the dtype of the data object
 * \param[in] dimensions the number of dimensions of the data object
 * \param[in] dims the actual of dimensions of the data object
 * \returns the size of a data object in bytes
 */
size_t data_size_in_bytes(pressio_dtype type, size_t const dimensions, size_t const dims[]);


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
  static pressio_data copy(const enum pressio_dtype dtype, const void* src, std::vector<size_t> const& dimensions) {
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
  static pressio_data copy(const enum pressio_dtype dtype, const void* src, size_t const num_dimensions, size_t const dimensions[]) {
    size_t bytes = data_size_in_bytes(dtype, num_dimensions, dimensions);
    void* data = nullptr;
    if(bytes != 0) {
      data = malloc(bytes);
      memcpy(data, src, bytes); 
    }
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
    size_t bytes = data_size_in_bytes(dtype, num_dimensions, dimensions);
    void* data = nullptr;
    if(bytes != 0) data = malloc(bytes);
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

  /**
   * clones a existing data buffer
   *
   * Does not use the copy constructor to enforce strict semantics around copies
   *
   * \param[in] src the object
   * \returns a new data structure
   *
   */
  static pressio_data clone(pressio_data const& src){
    size_t bytes = src.size_in_bytes(); 
    unsigned char* data = nullptr;
    if(bytes != 0 && src.data() != nullptr) {
      data = static_cast<unsigned char*>(malloc(bytes));
      memcpy(data, src.data(), src.size_in_bytes());
    }
    return pressio_data(src.dtype(),
        data,
        nullptr,
        pressio_data_libc_free_fn,
        src.num_dimensions(),
        src.dimensions().data()
        );
  }

  pressio_data() :
    data_dtype(pressio_byte_dtype),
    data_ptr(nullptr),
    metadata_ptr(nullptr),
    deleter(nullptr),
    dims(),
    capacity(0)
  {}

  ~pressio_data() {
    if(deleter!=nullptr) deleter(data_ptr,metadata_ptr);
  };

  /**copy-assignment, clones the data
   * \param[in] rhs the data to clone
   * \see pressio_data::clone
   * */
  pressio_data& operator=(pressio_data const& rhs) {
    if(this == &rhs) return *this;
    data_dtype = rhs.data_dtype;
    if(rhs.has_data() && rhs.size_in_bytes() > 0) {
      if(deleter != nullptr) deleter(data_ptr, metadata_ptr);
      data_ptr = malloc(rhs.size_in_bytes());
      memcpy(data_ptr, rhs.data_ptr, rhs.size_in_bytes());
    } else {
      data_ptr = nullptr;
    }
    metadata_ptr = nullptr;
    deleter = pressio_data_libc_free_fn;
    dims = rhs.dims;
    capacity = data_size_in_bytes(rhs.dtype(), rhs.dims.size(), rhs.dims.data()); //we only malloc size_in_bytes()
    return *this;
  }
  /**copy-constructor, clones the data
   * \param[in] rhs the data to clone
   * \see pressio_data::clone
   * */
  pressio_data(pressio_data const& rhs): 
    data_dtype(rhs.data_dtype),
    data_ptr((rhs.has_data())? malloc(rhs.size_in_bytes()) : nullptr),
    metadata_ptr(nullptr),
    deleter(pressio_data_libc_free_fn),
    dims(rhs.dims),
    capacity(data_size_in_bytes(rhs.dtype(), rhs.dims.size(), rhs.dims.data())) //we only malloc size_in_bytes
  {
    if(rhs.has_data() && rhs.size_in_bytes() > 0) {
      memcpy(data_ptr, rhs.data_ptr, rhs.size_in_bytes());
    }
  }
  /**
   * move-constructor
   *
   * \param[in] rhs the data buffer to move from
   * \returns a reference to the object moved into
   */
  pressio_data(pressio_data&& rhs) noexcept:
    data_dtype(rhs.data_dtype),
    data_ptr(compat::exchange(rhs.data_ptr, nullptr)),
    metadata_ptr(compat::exchange(rhs.metadata_ptr, nullptr)),
    deleter(compat::exchange(rhs.deleter, nullptr)),
    dims(compat::exchange(rhs.dims, {})),
    capacity(compat::exchange(rhs.capacity, 0)) //we take ownership, so take everything
    {}
  
  /**
   * move-assignment operator
   *
   * \param[in] rhs the data buffer to move from
   * \returns a l-value reference to the object moved into
   */
  pressio_data& operator=(pressio_data && rhs) noexcept {
    if(this==&rhs) return *this;
    if(deleter!=nullptr) deleter(data_ptr,metadata_ptr);
    data_dtype = rhs.data_dtype,
    data_ptr = compat::exchange(rhs.data_ptr, nullptr),
    metadata_ptr = compat::exchange(rhs.metadata_ptr, nullptr),
    deleter = compat::exchange(rhs.deleter, nullptr),
    dims = compat::exchange(rhs.dims, {});
    capacity = compat::exchange(rhs.capacity, 0);
    return *this;
  }

  /**
   * construct a literal pressio_data object from a initializer list
   *
   * \param[in] il initializer list to use to create the data object
   */
  template <class T>
  pressio_data(std::initializer_list<T> il):
    data_dtype(pressio_dtype_from_type<T>()),
    data_ptr((il.size() == 0)? nullptr:malloc(il.size() * sizeof(T))),
    metadata_ptr(nullptr),
    deleter(pressio_data_libc_free_fn),
    dims({il.size()}) 
  {
    std::copy(std::begin(il), std::end(il), static_cast<T*>(data_ptr));
  }
    

  /**
   * \returns a non-owning pointer to the data
   */
  void* data() const {
    return data_ptr;
  }

  /**
   * \returns true if the structure has has data
   */
  bool has_data() const {
    return data_ptr != nullptr && size_in_bytes() > 0;
  }
  
  /**
   * \returns the data type of the buffer
   */
  pressio_dtype dtype() const {
    return data_dtype;
  }

  /**
   * \param[in] dtype the data type for the buffer
   */
  void set_dtype(pressio_dtype dtype) {
    data_dtype = dtype;
  }

  /**
   * \param[in] dtype the new datatype to assign
   * \returns a new pressio_data structure based on the current structure with the new type
   */
  pressio_data cast(pressio_dtype dtype) const; 
  /**
   * \returns the number of dimensions
   */
  size_t num_dimensions() const {
    return dims.size();
  }

  /**
   * \returns a copy of the vector of dimensions
   */
  std::vector<size_t> const& dimensions() const {
    return dims;
  }

  /**
   * returns the dimensions normalized to remove 1's
   */
  std::vector<size_t> normalized_dims() const {
    std::vector<size_t> real_dims;
    std::copy_if(dims.begin(), dims.end(), std::back_inserter(real_dims), [](size_t i){ return i > 1; });
    return real_dims;
  }


  /**
   * changes the dimensions of the size of the memory
   * if the resulting buffer is smaller than the current buffer capacity in bytes, nothing else is done
   * if the resulting buffer is larger than the current buffer capacity in bytes, a realloc-like operation is performed
   * 
   * \param[in] dims the new dimensions to use
   * \returns the size of the new buffer in bytes, returns 0 if malloc fails
   *
   */
  size_t set_dimensions(std::vector<size_t>&& dims) {
    size_t new_size = data_size_in_bytes(data_dtype, dims.size(), dims.data());
    if(capacity_in_bytes() < new_size) {
      void* tmp = malloc(new_size);
      if(tmp == nullptr) {
        return 0;
      } else {
        memcpy(tmp, data_ptr, size_in_bytes());
        if(deleter!=nullptr) deleter(data_ptr,metadata_ptr);

        data_ptr = tmp;
        deleter = pressio_data_libc_free_fn;
        metadata_ptr = nullptr;
        capacity = new_size;
      }
    } 
    this->dims = std::move(dims);
    return size_in_bytes();
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
   * \returns the capacity of the buffer in bytes
   */
  size_t capacity_in_bytes() const {
    return capacity;
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


  /**
   * modifies the dimensions of this pressio_data structure in-place.
   *
   * This API does not change the size of the underlying buffer.
   * A future version of this API may change this.
   *
   * \param[in] new_dimensions the new dimensions to use
   *
   * \returns 0 if the resize was successful, negative values on warnings (i.e. dimensions mismatch), positive values on errors
   */
  int reshape(std::vector<size_t> const& new_dimensions) {
    const size_t old_size = data_size_in_elements(num_dimensions(), dims.data());
    const size_t new_size = data_size_in_elements(new_dimensions.size(), new_dimensions.data());

    dims = new_dimensions;

    if(old_size == new_size) {
      return 0;
    } else if (old_size > new_size){
      return -1;
    } else {
      return 1;
    }
  }

  /**
   * convert a pressio_data structure into a 1d c++ standard vector.  If the type doesn't match, it will be casted first
   * \returns the vector containing the data
   * \see pressio_data::cast
   */
  template <class T>
  std::vector<T> to_vector() const {
    if(pressio_dtype_from_type<T>() == dtype()) {
      return std::vector<T>(static_cast<T*>(data()), static_cast<T*>(data()) + num_elements());
    } else {
      auto casted = cast(pressio_dtype_from_type<T>());
      return std::vector<T>(static_cast<T*>(casted.data()), static_cast<T*>(casted.data()) + casted.num_elements());
    }
  }

  /**
   * convert a iterable type into a pressio_data object
   * \param[in] begin iterator to the beginning of the data
   * \param[in] end iterator to the end of the data
   * \returns a new 1d pressio_data object or matching type
   */
  template <class ForwardIt>
  pressio_data(ForwardIt begin, ForwardIt end):
    data_dtype(pressio_dtype_from_type<typename std::iterator_traits<ForwardIt>::value_type>()),
    data_ptr(malloc(std::distance(begin, end)*pressio_dtype_size(data_dtype))),
    metadata_ptr(nullptr),
    deleter(pressio_data_libc_free_fn),
    dims({static_cast<size_t>(std::distance(begin, end))})
  {
  using out_t = typename std::add_pointer<typename std::decay<
    typename std::iterator_traits<ForwardIt>::value_type>::type>::type;

    std::copy(begin, end, static_cast<out_t>(data_ptr));
  }

  /**
   * Permutes the dimensions of an array
   *
   * \param[in] axis by default reverses the axis, otherwise permutes the axes according to the dimensions
   * \returns the data with its axes permuted
   */
  pressio_data transpose(std::vector<size_t> const& axis = {}) const;
  
  /** 
   * \param[in] rhs the object to compare against
   * \returns true if the options are equal */
  bool operator==(pressio_data const& rhs) const;

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
      size_t const dimensions[]
      ):
    data_dtype(dtype),
    data_ptr(data),
    metadata_ptr(metadata),
    deleter(deleter),
    dims(dimensions, dimensions+num_dimensions),
    capacity(data_size_in_bytes(dtype, num_dimensions, dimensions))
  {}
  /**
   * constructor use the static methods instead
   * \param dtype the type of the data
   * \param data the buffer to use
   * \param metadata the meta data to pass to the deleter function
   * \param num_dimensions the number of dimensions to represent
   * \param dimensions of the data
   * \param capacity of the data
   */
  pressio_data(const pressio_dtype dtype,
      void* data,
      void* metadata,
      void (*deleter)(void*, void*),
      size_t const num_dimensions,
      size_t const dimensions[],
      size_t capacity
      ):
    data_dtype(dtype),
    data_ptr(data),
    metadata_ptr(metadata),
    deleter(deleter),
    dims(dimensions, dimensions+num_dimensions),
    capacity(capacity)
  {}
  pressio_dtype data_dtype;
  void* data_ptr;
  void* metadata_ptr;
  void (*deleter)(void*, void*);
  std::vector<size_t> dims;
  size_t capacity;
};

/**
 * get beginning and end pointers for two input data values
 *
 * \param[in] data first input data set
 * \param[in] f templated function to call, it must return the same type regardless of the type of the inputs.
 *            it should have the signature \code template <class T> f(T* input_begin, T* input_end) \endcode
 */
template <class ReturnType, class Function>
ReturnType pressio_data_for_each(pressio_data const& data, Function&& f)
{
  switch(data.dtype())
  {
    case pressio_double_dtype: 
      return std::forward<Function>(f)(
          static_cast<double*>(data.data()),
          static_cast<double*>(data.data()) + data.num_elements()
        );
    case pressio_float_dtype:
      return std::forward<Function>(f)(
          static_cast<float*>(data.data()),
          static_cast<float*>(data.data()) + data.num_elements()
        );
    case pressio_uint8_dtype:
      return std::forward<Function>(f)(
          static_cast<uint8_t*>(data.data()),
          static_cast<uint8_t*>(data.data()) + data.num_elements()
        );
    case pressio_uint16_dtype:
      return std::forward<Function>(f)(
          static_cast<uint16_t*>(data.data()),
          static_cast<uint16_t*>(data.data()) + data.num_elements()
        );
    case pressio_uint32_dtype:
      return std::forward<Function>(f)(
          static_cast<uint32_t*>(data.data()),
          static_cast<uint32_t*>(data.data()) + data.num_elements()
        );
    case pressio_uint64_dtype:
      return std::forward<Function>(f)(
          static_cast<uint64_t*>(data.data()),
          static_cast<uint64_t*>(data.data()) + data.num_elements()
        );
    case pressio_int8_dtype:
      return std::forward<Function>(f)(
          static_cast<int8_t*>(data.data()),
          static_cast<int8_t*>(data.data()) + data.num_elements()
        );
    case pressio_int16_dtype:
      return std::forward<Function>(f)(
          static_cast<int16_t*>(data.data()),
          static_cast<int16_t*>(data.data()) + data.num_elements()
        );
    case pressio_int32_dtype:
      return std::forward<Function>(f)(
          static_cast<int32_t*>(data.data()),
          static_cast<int32_t*>(data.data()) + data.num_elements()
        );
    case pressio_int64_dtype:
      return std::forward<Function>(f)(
          static_cast<int64_t*>(data.data()),
          static_cast<int64_t*>(data.data()) + data.num_elements()
        );
    case pressio_byte_dtype:
    default:
      return std::forward<Function>(f)(
          static_cast<unsigned char*>(data.data()),
          static_cast<unsigned char*>(data.data()) + data.size_in_bytes()
        );
  }
}

/**
 * get beginning and end pointers for two input data values
 *
 * \param[in] data first input data set
 * \param[in] f templated function to call, it must return the same type regardless of the type of the inputs.
 *            it should have the signature \code template <class T> f(T* input_begin, T* input_end) \endcode
 */
template <class ReturnType, class Function>
ReturnType pressio_data_for_each(pressio_data& data, Function&& f)
{
  switch(data.dtype())
  {
    case pressio_double_dtype: 
      return std::forward<Function>(f)(
          static_cast<double*>(data.data()),
          static_cast<double*>(data.data()) + data.num_elements()
        );
    case pressio_float_dtype:
      return std::forward<Function>(f)(
          static_cast<float*>(data.data()),
          static_cast<float*>(data.data()) + data.num_elements()
        );
    case pressio_uint8_dtype:
      return std::forward<Function>(f)(
          static_cast<uint8_t*>(data.data()),
          static_cast<uint8_t*>(data.data()) + data.num_elements()
        );
    case pressio_uint16_dtype:
      return std::forward<Function>(f)(
          static_cast<uint16_t*>(data.data()),
          static_cast<uint16_t*>(data.data()) + data.num_elements()
        );
    case pressio_uint32_dtype:
      return std::forward<Function>(f)(
          static_cast<uint32_t*>(data.data()),
          static_cast<uint32_t*>(data.data()) + data.num_elements()
        );
    case pressio_uint64_dtype:
      return std::forward<Function>(f)(
          static_cast<uint64_t*>(data.data()),
          static_cast<uint64_t*>(data.data()) + data.num_elements()
        );
    case pressio_int8_dtype:
      return std::forward<Function>(f)(
          static_cast<int8_t*>(data.data()),
          static_cast<int8_t*>(data.data()) + data.num_elements()
        );
    case pressio_int16_dtype:
      return std::forward<Function>(f)(
          static_cast<int16_t*>(data.data()),
          static_cast<int16_t*>(data.data()) + data.num_elements()
        );
    case pressio_int32_dtype:
      return std::forward<Function>(f)(
          static_cast<int32_t*>(data.data()),
          static_cast<int32_t*>(data.data()) + data.num_elements()
        );
    case pressio_int64_dtype:
      return std::forward<Function>(f)(
          static_cast<int64_t*>(data.data()),
          static_cast<int64_t*>(data.data()) + data.num_elements()
        );
    case pressio_byte_dtype:
    default:
      return std::forward<Function>(f)(
          static_cast<unsigned char*>(data.data()),
          static_cast<unsigned char*>(data.data()) + data.size_in_bytes()
        );
  }
}

namespace {
  template <class ReturnType, class Function, class Type1, class Type2>
  ReturnType pressio_data_for_each_call(pressio_data const& data, pressio_data const& data2, Function&& f) {
      return std::forward<Function>(f)(
          static_cast<Type1*>(data.data()),
          static_cast<Type1*>(data.data()) + data.num_elements(),
          static_cast<Type2*>(data2.data()),
          static_cast<Type2*>(data2.data()) + data2.num_elements()
        );
  }
  template <class ReturnType, class Function, class Type1>
  ReturnType pressio_data_for_each_type2_switch(pressio_data const& data, pressio_data const& data2, Function&& f) {
    switch(data2.dtype()) {
    case pressio_double_dtype: 
      return pressio_data_for_each_call<ReturnType, Function, Type1, double>(data, data2, std::forward<Function>(f));
    case pressio_float_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, float>(data, data2, std::forward<Function>(f));
    case pressio_uint8_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint8_t>(data, data2, std::forward<Function>(f));
    case pressio_uint16_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint16_t>(data, data2, std::forward<Function>(f));
    case pressio_uint32_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint32_t>(data, data2, std::forward<Function>(f));
    case pressio_uint64_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint64_t>(data, data2, std::forward<Function>(f));
    case pressio_int8_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int8_t>(data, data2, std::forward<Function>(f));
    case pressio_int16_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int16_t>(data, data2, std::forward<Function>(f));
    case pressio_int32_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int32_t>(data, data2, std::forward<Function>(f));
    case pressio_int64_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int64_t>(data, data2, std::forward<Function>(f));
    default:
      return pressio_data_for_each_call<ReturnType, Function, Type1, char>(data, data2, std::forward<Function>(f));
    }
  }
  template <class ReturnType, class Function, class Type1, class Type2>
  ReturnType pressio_data_for_each_call(pressio_data& data, pressio_data& data2, Function&& f) {
      return std::forward<Function>(f)(
          static_cast<Type1*>(data.data()),
          static_cast<Type1*>(data.data()) + data.num_elements(),
          static_cast<Type2*>(data2.data())
        );
  }
  template <class ReturnType, class Function, class Type1>
  ReturnType pressio_data_for_each_type2_switch(pressio_data& data, pressio_data& data2, Function&& f) {
    switch(data2.dtype()) {
    case pressio_double_dtype: 
      return pressio_data_for_each_call<ReturnType, Function, Type1, double>(data, data2, std::forward<Function>(f));
    case pressio_float_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, float>(data, data2, std::forward<Function>(f));
    case pressio_uint8_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint8_t>(data, data2, std::forward<Function>(f));
    case pressio_uint16_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint16_t>(data, data2, std::forward<Function>(f));
    case pressio_uint32_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint32_t>(data, data2, std::forward<Function>(f));
    case pressio_uint64_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, uint64_t>(data, data2, std::forward<Function>(f));
    case pressio_int8_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int8_t>(data, data2, std::forward<Function>(f));
    case pressio_int16_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int16_t>(data, data2, std::forward<Function>(f));
    case pressio_int32_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int32_t>(data, data2, std::forward<Function>(f));
    case pressio_int64_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, int64_t>(data, data2, std::forward<Function>(f));
    default:
      return pressio_data_for_each_call<ReturnType, Function, Type1, char>(data, data2, std::forward<Function>(f));
    }
  }
}


/**
 * get beginning and end pointers for two input data values
 *
 * \param[in] data first input data set
 * \param[in] data2 second input data set
 * \param[in] f templated function to call, it must return the same type regardless of the type of the inputs.
 *            it should have the signature \code template <class T, class U> ReturnType f(T* input_begin, T* input_end, T* input2_begin)  where U is some type\endcode
 */
template <class ReturnType, class Function>
ReturnType pressio_data_for_each(pressio_data const& data, pressio_data const& data2, Function&& f) 
{
    switch(data.dtype()) {
    case pressio_double_dtype: 
      return pressio_data_for_each_type2_switch<ReturnType, Function, double>(data, data2, std::forward<Function>(f));
    case pressio_float_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, float>(data, data2, std::forward<Function>(f));
    case pressio_uint8_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint8_t>(data, data2, std::forward<Function>(f));
    case pressio_uint16_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint16_t>(data, data2, std::forward<Function>(f));
    case pressio_uint32_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint32_t>(data, data2, std::forward<Function>(f));
    case pressio_uint64_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint64_t>(data, data2, std::forward<Function>(f));
    case pressio_int8_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int8_t>(data, data2, std::forward<Function>(f));
    case pressio_int16_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int16_t>(data, data2, std::forward<Function>(f));
    case pressio_int32_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int32_t>(data, data2, std::forward<Function>(f));
    case pressio_int64_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int64_t>(data, data2, std::forward<Function>(f));
    default:
      return pressio_data_for_each_type2_switch<ReturnType, Function, char>(data, data2, std::forward<Function>(f));
    }
}
/**
 * get beginning and end pointers for two input data values
 *
 * \param[in] data first input data set
 * \param[in] data2 second input data set
 * \param[in] f templated function to call, it must return the same type regardless of the type of the inputs.
 *            it should have the signature \code template <class T, class U> ReturnType f(T* input_begin, T* input_end, T* input2_begin)  where U is some type\endcode
 */
template <class ReturnType, class Function>
ReturnType pressio_data_for_each(pressio_data& data, pressio_data& data2, Function&& f) 
{
    switch(data.dtype()) {
    case pressio_double_dtype: 
      return pressio_data_for_each_type2_switch<ReturnType, Function, double>(data, data2, std::forward<Function>(f));
    case pressio_float_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, float>(data, data2, std::forward<Function>(f));
    case pressio_uint8_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint8_t>(data, data2, std::forward<Function>(f));
    case pressio_uint16_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint16_t>(data, data2, std::forward<Function>(f));
    case pressio_uint32_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint32_t>(data, data2, std::forward<Function>(f));
    case pressio_uint64_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, uint64_t>(data, data2, std::forward<Function>(f));
    case pressio_int8_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int8_t>(data, data2, std::forward<Function>(f));
    case pressio_int16_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int16_t>(data, data2, std::forward<Function>(f));
    case pressio_int32_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int32_t>(data, data2, std::forward<Function>(f));
    case pressio_int64_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, int64_t>(data, data2, std::forward<Function>(f));
    default:
      return pressio_data_for_each_type2_switch<ReturnType, Function, char>(data, data2, std::forward<Function>(f));
    }
}

#endif /* end of include guard: PRESSIO_DATA_CPP_H */
