#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIBLOSSY_DTYPE
#define LIBLOSSY_DTYPE


/*! \file  
 *  \brief Information on types used by liblossy
 */

/**
 * data types recognized by liblossy for compression and decompression
 */
enum lossy_dtype {
  /** 64 bit double precision floating point */ lossy_double_dtype, 
  /** 32 bit double precision floating point */lossy_float_dtype,
  /**  8 bit unsigned integer */ lossy_uint8_dtype,
  /** 16 bit unsigned integer */ lossy_uint16_dtype,
  /** 32 bit unsigned integer */ lossy_uint32_dtype,
  /** 64 bit unsigned integer */ lossy_uint64_dtype,
  /**  8 bit signed integer */lossy_int8_dtype,
  /** 16 bit signed integer */lossy_int16_dtype,
  /** 32 bit signed integer */lossy_int32_dtype,
  /** 64 bit signed integer */lossy_int64_dtype,
  /** 8 bit data type */ lossy_byte_dtype,
};

/**
 * \returns the size in bytes of a liblossy recognized type
 */
int lossy_dtype_size (enum lossy_dtype dtype);

#endif

#ifdef __cplusplus
}
#endif
