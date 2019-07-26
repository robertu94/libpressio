#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
#include <cstring>
#include "pressio_data.h"

void pressio_data_libc_free_fn(void* data, void*) {
  free(data);
}

namespace {
  template <class T>
  size_t data_size_in_bytes(pressio_dtype type, size_t dimentions, T const dims[]) {
    size_t totalsize = 1;
    for (size_t i = 0; i < dimentions; ++i) {
      totalsize *= dims[i];
    }
    return totalsize * pressio_dtype_size(type);
  }
}

struct pressio_data {
  static pressio_data empty(const pressio_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
    return pressio_data(dtype, nullptr, nullptr, nullptr, num_dimentions, dimentions);
  }

  static pressio_data nonowning(const pressio_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[]) {
    return pressio_data(dtype, data, nullptr, nullptr, num_dimentions, dimentions);
  }

  static pressio_data owning(const pressio_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
    void* data = malloc(data_size_in_bytes(dtype, num_dimentions, dimentions));
    return pressio_data(dtype, data, nullptr, pressio_data_libc_free_fn, num_dimentions, dimentions);
  }
  static pressio_data move(const pressio_dtype dtype,
      void* data,
      size_t const num_dimentions,
      size_t const dimentions[],
      pressio_data_delete_fn deleter,
      void* metadata) {
    return pressio_data(dtype, data, metadata, deleter, num_dimentions, dimentions);
  }



  ~pressio_data() {
    if(deleter!=nullptr) deleter(data_ptr,metadata_ptr);
  };

  //disable copy constructors/assignment to handle case of non-owning pointers
  pressio_data& operator=(pressio_data const& rhs)=delete;
  pressio_data(pressio_data const& rhs)=delete; 

  pressio_data(pressio_data&& rhs):
    data_dtype(rhs.data_dtype),
    data_ptr(std::exchange(rhs.data_ptr, nullptr)),
    metadata_ptr(std::exchange(rhs.metadata_ptr, nullptr)),
    deleter(std::exchange(rhs.deleter, nullptr)),
    dims(std::exchange(rhs.dims, {})) {}
  
  pressio_data& operator=(pressio_data && rhs) {
    data_dtype = rhs.data_dtype,
    data_ptr = std::exchange(rhs.data_ptr, nullptr),
    metadata_ptr = std::exchange(rhs.metadata_ptr, nullptr),
    deleter = std::exchange(rhs.deleter, nullptr),
    dims = std::exchange(rhs.dims, {});
    return *this;
  }

  void* data() const {
    return data_ptr;
  }
  
  pressio_dtype dtype() const {
    return data_dtype;
  }

  size_t dimentions() const {
    return dims.size();
  }

  size_t get_dimention(size_t idx) const {
    if(idx >= dimentions()) return 0;
    else return dims[idx];
  }

  size_t size_in_bytes() const {
    return data_size_in_bytes(data_dtype, dimentions(), dims.data());
  }

  private:
  pressio_data(const pressio_dtype dtype,
      void* data,
      void* metadata,
      void (*deleter)(void*, void*),
      size_t const num_dimentions,
      size_t const dimentions[]):
    data_dtype(dtype),
    data_ptr(data),
    metadata_ptr(metadata),
    deleter(deleter),
    dims(dimentions, dimentions+num_dimentions)
  {}
  pressio_dtype data_dtype;
  void* data_ptr;
  void* metadata_ptr;
  void (*deleter)(void*, void*);
  std::vector<size_t> dims;
};

extern "C" {

struct pressio_data* pressio_data_new_move(const enum pressio_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[], pressio_data_delete_fn deleter, void* metadata) {
  return new pressio_data(pressio_data::move(dtype, data, num_dimentions, dimentions, deleter, metadata));
}

struct pressio_data* pressio_data_new_copy(const enum pressio_dtype dtype, void* src, size_t const num_dimentions, size_t const dimentions[]) {
  size_t bytes = data_size_in_bytes(dtype, num_dimentions, dimentions);
  void* data = malloc(bytes);
  memcpy(data, src, bytes);
  return new pressio_data(pressio_data::move(dtype, data, num_dimentions, dimentions, pressio_data_libc_free_fn, nullptr));
}

struct pressio_data* pressio_data_new_owning(const enum pressio_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
  return new pressio_data(pressio_data::owning(dtype, num_dimentions, dimentions));
}

struct pressio_data* pressio_data_new_nonowning(const enum pressio_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[]) {
  return new pressio_data(pressio_data::nonowning(dtype, data, num_dimentions, dimentions));
}

struct pressio_data* pressio_data_new_empty(const pressio_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
  return new pressio_data(pressio_data::empty(dtype, num_dimentions, dimentions));
}

void pressio_data_free(struct pressio_data* data) {
  delete data;
}

void* pressio_data_copy(struct pressio_data const* data, size_t* out_bytes) {
  if(out_bytes != nullptr) {
    *out_bytes = data->size_in_bytes();
  } 

  void* copy = malloc(data->size_in_bytes());
  memcpy(copy, data->data(), data->size_in_bytes());
  return copy;
}

void* pressio_data_ptr(struct pressio_data const* data, size_t* out_bytes) {
  if(out_bytes != nullptr) *out_bytes = data->size_in_bytes();
  return data->data();
}

pressio_dtype pressio_data_dtype(struct pressio_data const* data) {
  return data->dtype();
}

bool pressio_data_has_data(struct pressio_data const* data) {
  return data->data() == nullptr;
}

size_t pressio_data_num_dimentions(struct pressio_data const* data) {
  return data->dimentions();
}

size_t pressio_data_get_dimention(struct pressio_data const* data, size_t dimension) {
  return data->get_dimention(dimension);
}

}
