#include <cstdint>
#include "lossy_dtype.h"

extern "C" {
int lossy_dtype_size (enum lossy_dtype dtype) {
  switch(dtype) {
    case lossy_double_dtype:
      return sizeof(double);
    case lossy_float_dtype:
      return sizeof(float);
    case lossy_uint8_dtype:
      return sizeof(uint8_t);
    case lossy_uint16_dtype:
      return sizeof(uint16_t);
    case lossy_uint32_dtype:
      return sizeof(uint32_t);
    case lossy_uint64_dtype:
      return sizeof(uint64_t);
    case lossy_int8_dtype:
      return sizeof(int8_t);
    case lossy_int16_dtype:
      return sizeof(int16_t);
    case lossy_int32_dtype:
      return sizeof(int32_t);
    case lossy_int64_dtype:
      return sizeof(int64_t);
    case lossy_byte_dtype:
      return sizeof(unsigned char);
    default:
      return -1;
  }
}
}
