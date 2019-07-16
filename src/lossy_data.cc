#include <cstdlib>
#include <cstring>
#include <vector>
#include "lossy_data.h"

struct lossy_data {
  lossy_data(const lossy_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[]):
    data_dtype(dtype), data_ptr(data), dims(dimentions, dimentions+num_dimentions) {}

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
    auto dims = dimentions();
    size_t totalsize = 1;
    for (decltype(dims) i = 0; i < dims; ++i) {
      totalsize *= get_dimention(i);
    }
    return totalsize * lossy_dtype_size(data_dtype);
  }

  private:
  lossy_dtype data_dtype;
  void* data_ptr;
  std::vector<size_t> dims;
};

extern "C" {

struct lossy_data* lossy_data_new(const lossy_dtype dtype, void* data, size_t const num_dimentions, size_t const dimentions[]) {
  return new lossy_data(dtype, data, num_dimentions, dimentions);
}

struct lossy_data* lossy_data_new_empty(const lossy_dtype dtype, size_t const num_dimentions, size_t const dimentions[]) {
  return new lossy_data(dtype, nullptr, num_dimentions, dimentions);
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
