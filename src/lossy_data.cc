#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
#include <cstring>
#include "lossy_data.h"

void lossy_data_libc_free_fn(void* data, void*) {
  free(data);
}

namespace {
  template <class T>
  size_t data_size_in_bytes(lossy_dtype type, size_t dimentions, T const dims[]) {
    size_t totalsize = 1;
    for (size_t i = 0; i < dimentions; ++i) {
      totalsize *= dims[i];
    }
    return totalsize * lossy_dtype_size(type);
  }
}

struct lossy_data {
  static lossy_data empty(const lossy_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
    return lossy_data(dtype, nullptr, nullptr, nullptr, num_dimentions, dimentions);
  }

  static lossy_data nonowning(const lossy_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[]) {
    return lossy_data(dtype, data, nullptr, nullptr, num_dimentions, dimentions);
  }

  static lossy_data owning(const lossy_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
    void* data = malloc(data_size_in_bytes(dtype, num_dimentions, dimentions));
    return lossy_data(dtype, data, nullptr, lossy_data_libc_free_fn, num_dimentions, dimentions);
  }
  static lossy_data move(const lossy_dtype dtype,
      void* data,
      size_t const num_dimentions,
      size_t const dimentions[],
      lossy_data_delete_fn deleter,
      void* metadata) {
    return lossy_data(dtype, data, metadata, deleter, num_dimentions, dimentions);
  }



  ~lossy_data() {
    if(deleter!=nullptr) deleter(data_ptr,metadata_ptr);
  };

  //disable copy constructors/assignment to handle case of non-owning pointers
  lossy_data& operator=(lossy_data const& rhs)=delete;
  lossy_data(lossy_data const& rhs)=delete; 

  lossy_data(lossy_data&& rhs):
    data_dtype(rhs.data_dtype),
    data_ptr(std::exchange(rhs.data_ptr, nullptr)),
    metadata_ptr(std::exchange(rhs.metadata_ptr, nullptr)),
    deleter(std::exchange(rhs.deleter, nullptr)),
    dims(std::exchange(rhs.dims, {})) {}
  
  lossy_data& operator=(lossy_data && rhs) {
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
  
  lossy_dtype dtype() const {
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
  lossy_data(const lossy_dtype dtype,
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
  lossy_dtype data_dtype;
  void* data_ptr;
  void* metadata_ptr;
  void (*deleter)(void*, void*);
  std::vector<size_t> dims;
};

extern "C" {

struct lossy_data* lossy_data_new_move(const enum lossy_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[], lossy_data_delete_fn deleter, void* metadata) {
  return new lossy_data(lossy_data::move(dtype, data, num_dimentions, dimentions, deleter, metadata));
}

struct lossy_data* lossy_data_new_copy(const enum lossy_dtype dtype, void* src, size_t const num_dimentions, size_t const dimentions[]) {
  size_t bytes = data_size_in_bytes(dtype, num_dimentions, dimentions);
  void* data = malloc(bytes);
  memcpy(data, src, bytes);
  return new lossy_data(lossy_data::move(dtype, data, num_dimentions, dimentions, lossy_data_libc_free_fn, nullptr));
}

struct lossy_data* lossy_data_new_owning(const enum lossy_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
  return new lossy_data(lossy_data::owning(dtype, num_dimentions, dimentions));
}

struct lossy_data* lossy_data_new_nonowning(const enum lossy_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[]) {
  return new lossy_data(lossy_data::nonowning(dtype, data, num_dimentions, dimentions));
}

struct lossy_data* lossy_data_new_empty(const lossy_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
  return new lossy_data(lossy_data::empty(dtype, num_dimentions, dimentions));
}

void lossy_data_free(struct lossy_data* data) {
  delete data;
}

void* lossy_data_copy(struct lossy_data const* data, size_t* out_bytes) {
  if(out_bytes != nullptr) {
    *out_bytes = data->size_in_bytes();
  } 

  void* copy = malloc(data->size_in_bytes());
  memcpy(copy, data->data(), data->size_in_bytes());
  return copy;
}

void* lossy_data_ptr(struct lossy_data const* data, size_t* out_bytes) {
  if(out_bytes != nullptr) *out_bytes = data->size_in_bytes();
  return data->data();
}

lossy_dtype lossy_data_dtype(struct lossy_data const* data) {
  return data->dtype();
}

bool lossy_data_has_data(struct lossy_data const* data) {
  return data->data() == nullptr;
}

size_t lossy_data_num_dimentions(struct lossy_data const* data) {
  return data->dimentions();
}

size_t lossy_data_get_dimention(struct lossy_data const* data, size_t dimension) {
  return data->get_dimention(dimension);
}

}
