#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstring>
#include "pressio_data.h"
#include "libpressio_ext/cpp/data.h"

void pressio_data_libc_free_fn(void* data, void*) {
  free(data);
}


extern "C" {

struct pressio_data* pressio_data_new_move(const enum pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[], pressio_data_delete_fn deleter, void* metadata) {
  return new pressio_data(pressio_data::move(dtype, data, num_dimensions, dimensions, deleter, metadata));
}

struct pressio_data* pressio_data_new_copy(const enum pressio_dtype dtype, void* src, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::copy(dtype, src, num_dimensions, dimensions));
}

struct pressio_data* pressio_data_new_owning(const enum pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::owning(dtype, num_dimensions, dimensions));
}

struct pressio_data* pressio_data_new_nonowning(const enum pressio_dtype dtype, void* data, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::nonowning(dtype, data, num_dimensions, dimensions));
}

struct pressio_data* pressio_data_new_empty(const pressio_dtype dtype, size_t const num_dimensions, size_t const dimensions[]) {
  return new pressio_data(pressio_data::empty(dtype, num_dimensions, dimensions));
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
  return data->data() != nullptr;
}

size_t pressio_data_num_dimensions(struct pressio_data const* data) {
  return data->dimensions();
}

size_t pressio_data_get_dimension(struct pressio_data const* data, size_t dimension) {
  return data->get_dimension(dimension);
}

size_t pressio_data_get_bytes(struct pressio_data const* data) {
  return data->size_in_bytes();
}

size_t pressio_data_num_elements(struct pressio_data const* data) {
  return data->num_elements();
}


}
