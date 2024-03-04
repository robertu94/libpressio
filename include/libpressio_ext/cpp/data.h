#ifndef PRESSIO_DATA_CPP_H
#define PRESSIO_DATA_CPP_H



#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <algorithm>
#include "pressio_data.h"
#include "memory.h"

#include "libpressio_ext/cpp/dtype.h"
#include "std_compat/optional.h"
#include <std_compat/memory.h>

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
  static pressio_data empty(const pressio_dtype dtype, std::vector<size_t> const& dimensions);
  /**  
   * creates a non-owning reference to data
   *
   * \param[in] dtype the type of the buffer
   * \param[in] data the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \returns an non-owning data object (i.e. calling pressio_data_free will not deallocate this memory)
   * \see pressio_data_new_nonowning
   * */
  static pressio_data nonowning(const pressio_dtype dtype, void* data, std::vector<size_t> const& dimensions);
  /**  
   * creates a non-owning reference to data
   *
   * \param[in] dtype the type of the buffer
   * \param[in] data the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \param[in] domain_id domain_id for the non-owning pointer
   * \returns an non-owning data object (i.e. calling pressio_data_free will not deallocate this memory)
   * \see pressio_data_new_nonowning
   * */
  static pressio_data nonowning(const pressio_dtype dtype, void* data, std::vector<size_t> const& dimensions, std::string const& domain_id);
  /**  
   * creates a copy of a data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] src the buffer to copy \param[in] dimensions the dimensions of the buffer \returns an owning copy of the data object \see pressio_data_new_copy */
  static pressio_data copy(const enum pressio_dtype dtype, const void* src, std::vector<size_t> const& dimensions);
  /**  
   * creates a copy of a data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_owning
   * */
  static pressio_data owning(const pressio_dtype dtype, std::vector<size_t> const& dimensions);
  /**  
   * creates a copy of a data buffer in the specified domain
   *
   * \param[in] dtype the type of the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \param[in] domain the dimensions of the buffer
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_owning
   * */
  static pressio_data owning(const pressio_dtype dtype, std::vector<size_t> const& dimensions, std::shared_ptr<pressio_domain> && domain);
  /**  
   * creates a copy of a data buffer in the specified domain
   *
   * \param[in] dtype the type of the buffer
   * \param[in] dimensions the dimensions of the buffer
   * \param[in] domain the dimensions of the buffer
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_owning
   * */
  static pressio_data owning(const pressio_dtype dtype, std::vector<size_t> const& dimensions, std::shared_ptr<pressio_domain> const& domain);
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
      void* metadata);

  /**  
   * allocates a new empty data buffer
   *
   * \param[in] dtype the type the buffer will contain
   * \param[in] num_dimensions the length of dimensions
   * \param[in] dimensions the dimensions of the expected buffer
   * \returns an empty data object (i.e. has no data)
   * \see pressio_data_new_empty
   * */
  static pressio_data empty(const pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]);

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
  static pressio_data nonowning(const pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[]);
  /**  
   * creates a non-owning reference to data
   *
   * \param[in] dtype the type of the buffer
   * \param[in] data the buffer
   * \param[in] num_dimensions the number of entries in dimensions
   * \param[in] dimensions the dimensions of the data
   * \param[in] domain_id the domain_id of the pointer
   * \returns an non-owning data object (i.e. calling pressio_data_free will not deallocate this memory)
   * \see pressio_data_new_nonowning
   * */
  static pressio_data nonowning(const pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[], std::string const& domain_id);

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
  static pressio_data copy(const enum pressio_dtype dtype, const void* src, size_t const num_dimensions, size_t const dimensions[]);

  /**  
   * creates a copy of a data buffer
   *
   * \param[in] dtype the type of the buffer
   * \param[in] num_dimensions the number of entries in dimensions
   * \param[in] dimensions the dimensions of the data
   * \returns an owning data object with uninitialized memory
   * \see pressio_data_new_owning
   * */
  static pressio_data owning(const pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]);


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
      void* metadata);

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
      pressio_memory cloned(src.memory);
      return pressio_data(src.dtype(), src.dimensions(), std::move(cloned));
  }

  pressio_data() :
    data_dtype(pressio_byte_dtype),
    dims(),
    memory()
  {}

  /**
   * construct a literal pressio_data object from a initializer list
   *
   * \param[in] il initializer list to use to create the data object
   */
  template <class T>
  pressio_data(std::initializer_list<T> il):
    data_dtype(pressio_dtype_from_type<T>()),
    dims({il.size()}),
    memory(il.size()*sizeof(T))
  {
    std::copy(std::begin(il), std::end(il), static_cast<T*>(memory.data()));
  }
    

  /**
   * \returns a non-owning pointer to the data
   */
  void* data() const {
    return memory.data();
  }

  /**
   * \returns true if the structure has has data
   */
  bool has_data() const {
    return memory.data() != nullptr && size_in_bytes() > 0;
  }
  
  /**
   * \returns the data type of the buffer
   */
  pressio_dtype dtype() const {
    return data_dtype;
  }

  std::shared_ptr<pressio_domain> domain() const {
      return memory.domain();
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
  std::vector<size_t> normalized_dims(compat::optional<size_t> n={}, size_t fill=0) const;


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
        pressio_memory new_mem(new_size, memory.domain());
        memory = std::move(new_mem);
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
    return memory.capacity();
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
   */
  template <class ForwardIt>
  pressio_data(ForwardIt begin, ForwardIt end):
    data_dtype(pressio_dtype_from_type<typename std::iterator_traits<ForwardIt>::value_type>()),
    dims({static_cast<size_t>(std::distance(begin, end))}),
    memory(std::distance(begin, end)*pressio_dtype_size(data_dtype))
  {
  using out_t = typename std::add_pointer<typename std::decay<
    typename std::iterator_traits<ForwardIt>::value_type>::type>::type;

    std::copy(begin, end, static_cast<out_t>(memory.data()));
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
  pressio_data(const pressio_dtype dtype,
      std::vector<size_t>const& dimensions,
      pressio_memory&& memory
      ):
    data_dtype(dtype),
    dims(dimensions),
    memory(std::move(memory))
  {
      if(memory.capacity() != 0 && size_in_bytes() > memory.capacity()) {
          throw std::runtime_error("pressio_data size exceeds capacity");
      }
  }
  pressio_dtype data_dtype;
  std::vector<size_t> dims;
  pressio_memory memory;
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
    case pressio_bool_dtype:
      return std::forward<Function>(f)(
          static_cast<bool*>(data.data()),
          static_cast<bool*>(data.data()) + data.num_elements()
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
    case pressio_bool_dtype:
      return std::forward<Function>(f)(
          static_cast<bool*>(data.data()),
          static_cast<bool*>(data.data()) + data.num_elements()
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
    case pressio_bool_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, bool>(data, data2, std::forward<Function>(f));
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
    case pressio_bool_dtype:
      return pressio_data_for_each_call<ReturnType, Function, Type1, bool>(data, data2, std::forward<Function>(f));
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
    case pressio_bool_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, bool>(data, data2, std::forward<Function>(f));
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
    case pressio_bool_dtype:
      return pressio_data_for_each_type2_switch<ReturnType, Function, bool>(data, data2, std::forward<Function>(f));
    default:
      return pressio_data_for_each_type2_switch<ReturnType, Function, char>(data, data2, std::forward<Function>(f));
    }
}

#endif /* end of include guard: PRESSIO_DATA_CPP_H */
