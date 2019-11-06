#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIBPRESSIO_DTYPE
#define LIBPRESSIO_DTYPE


/*! \file  
 *  \brief Information on types used by libpressio
 */

/**
 * data types recognized by libpressio for compression and decompression
 */
enum pressio_dtype {
  /** 64 bit double precision floating point */ pressio_double_dtype, 
  /** 32 bit double precision floating point */pressio_float_dtype,
  /**  8 bit unsigned integer */ pressio_uint8_dtype,
  /** 16 bit unsigned integer */ pressio_uint16_dtype,
  /** 32 bit unsigned integer */ pressio_uint32_dtype,
  /** 64 bit unsigned integer */ pressio_uint64_dtype,
  /**  8 bit signed integer */pressio_int8_dtype,
  /** 16 bit signed integer */pressio_int16_dtype,
  /** 32 bit signed integer */pressio_int32_dtype,
  /** 64 bit signed integer */pressio_int64_dtype,
  /** 8 bit data type */ pressio_byte_dtype,
};

/**
 * \returns the size in bytes of a libpressio recognized type
 */
int pressio_dtype_size (enum pressio_dtype dtype);

/**
 * \returns non-zero if the type is a floating point value
 */
int pressio_dtype_is_floating (enum pressio_dtype dtype);

/**
 * \returns non-zero if the type is a numeric value
 */
int pressio_dtype_is_numeric (enum pressio_dtype dtype);

/**
 * \returns non-zero if the type is signed
 */
int pressio_dtype_is_signed (enum pressio_dtype dtype);

#endif

#ifdef __cplusplus
}
#endif
