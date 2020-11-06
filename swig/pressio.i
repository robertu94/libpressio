/*
python bindings for pressio
*/

%module pressio

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_STRICT_BYTE_CHAR
#include "pressio.h"
#include "pressio_compressor.h"
#include "pressio_dtype.h"
#include "pressio_metrics.h"
#include "pressio_option.h"
#include "pressio_options.h"
#include "pressio_options_iter.h"
#include "pressio_data.h"
#include "libpressio_ext/io/pressio_io.h"
#include "libpressio_ext/io/posix.h"
#include "pypressio.h"
%}

%include <stdint.i>
%include "numpy.i"
%init %{
import_array();
%}


%include "pybuffer.i"
%pybuffer_string(const char* compressor_id)


%include <std_string.i>
%include <std_vector.i>
%include <cpointer.i>

%pointer_functions(int, int32)
%pointer_functions(unsigned int, uint32)
%pointer_functions(double, double)
%pointer_functions(float, float)

%include "pypressio.h"

%numpy_typemaps(float             , NPY_FLOAT    , size_t)
%numpy_typemaps(double            , NPY_DOUBLE   , size_t)
%numpy_typemaps(int8_t            , NPY_INT8     , size_t)
%numpy_typemaps(int16_t           , NPY_INT16    , size_t)
%numpy_typemaps(int32_t           , NPY_INT32    , size_t)
%numpy_typemaps(int64_t           , NPY_INT64    , size_t)
%numpy_typemaps(uint8_t           , NPY_UINT8    , size_t)
%numpy_typemaps(uint16_t          , NPY_UINT16   , size_t)
%numpy_typemaps(uint32_t          , NPY_UINT32   , size_t)
%numpy_typemaps(uint64_t          , NPY_UINT64   , size_t)
%numpy_typemaps(float             , NPY_FLOAT    , long int)
%numpy_typemaps(double            , NPY_DOUBLE   , long int)
%numpy_typemaps(int8_t            , NPY_INT8     , long int)
%numpy_typemaps(int16_t           , NPY_INT16    , long int)
%numpy_typemaps(int32_t           , NPY_INT32    , long int)
%numpy_typemaps(int64_t           , NPY_INT64    , long int)
%numpy_typemaps(uint8_t           , NPY_UINT8    , long int)
%numpy_typemaps(uint16_t          , NPY_UINT16   , long int)
%numpy_typemaps(uint32_t          , NPY_UINT32   , long int)
%numpy_typemaps(uint64_t          , NPY_UINT64   , long int)

%define pressio_numpy_type(type)

  %apply (type* INPLACE_ARRAY1, size_t DIM1 ) {( type * data, size_t r1)};
  %apply (type* INPLACE_ARRAY2, size_t DIM1, size_t DIM2 ) { ( type * data, size_t r1, size_t r2)};
  %apply (type* INPLACE_ARRAY3, size_t DIM1, size_t DIM2, size_t DIM3 ) {( type* data, size_t r1, size_t r2, size_t r3)};
  %apply (type* INPLACE_ARRAY4, size_t DIM1, size_t DIM2, size_t DIM3, size_t DIM4 ) {( type* data, size_t r1, size_t r2, size_t r3, size_t r4)};
  %apply (type** ARGOUTVIEWM_ARRAY1, long int* DIM1) {( type** ptr_argout, long int* r1)};
  %apply (type** ARGOUTVIEWM_ARRAY2, long int* DIM1, long int* DIM2) {( type** ptr_argout, long int* r1, long int* r2)};
  %apply (type** ARGOUTVIEWM_ARRAY3, long int* DIM1, long int* DIM2, long int* DIM3) {( type** ptr_argout, long int* r1, long int* r2, long int* r3)};
  %apply (type** ARGOUTVIEWM_ARRAY4, long int* DIM1, long int* DIM2, long int* DIM3, long int* DIM4) {( type** ptr_argout, long int* r1, long int* r2, long int* r3, long int* r4)};
namespace std {
  %template( vector_ ## type ) vector< type >;
}
  %template( _pressio_io_data_to_numpy_1d_ ## type ) _pressio_io_data_to_numpy_1d< type >;
  %template( _pressio_io_data_to_numpy_2d_ ## type ) _pressio_io_data_to_numpy_2d< type >;
  %template( _pressio_io_data_to_numpy_3d_ ## type ) _pressio_io_data_to_numpy_3d< type >;
  %template( _pressio_io_data_to_numpy_4d_ ## type ) _pressio_io_data_to_numpy_4d< type >;
  %template( _pressio_io_data_from_numpy_1d_ ## type ) _pressio_io_data_from_numpy_1d< type >;
  %template( _pressio_io_data_from_numpy_2d_ ## type ) _pressio_io_data_from_numpy_2d< type >;
  %template( _pressio_io_data_from_numpy_3d_ ## type ) _pressio_io_data_from_numpy_3d< type >;
  %template( _pressio_io_data_from_numpy_4d_ ## type ) _pressio_io_data_from_numpy_4d< type >;
%enddef

pressio_numpy_type(float);
pressio_numpy_type(double);
pressio_numpy_type(uint8_t);
pressio_numpy_type(uint16_t);
pressio_numpy_type(uint32_t);
pressio_numpy_type(uint64_t);
pressio_numpy_type(int8_t);
pressio_numpy_type(int16_t);
pressio_numpy_type(int32_t);
pressio_numpy_type(int64_t);

namespace std { 
  %template() vector<size_t>;
  %template(vector_string) vector<std::string>;
}

%rename("%(strip:[pressio_])s") "";

%pythoncode %{
import numpy
__pressio_from_numpy = {
  (1, numpy.dtype('float32')): _pressio_io_data_from_numpy_1d_float,
  (2, numpy.dtype('float32')): _pressio_io_data_from_numpy_2d_float,
  (3, numpy.dtype('float32')): _pressio_io_data_from_numpy_3d_float,
  (4, numpy.dtype('float32')): _pressio_io_data_from_numpy_4d_float,
  (1, numpy.dtype('float64')): _pressio_io_data_from_numpy_1d_double,
  (2, numpy.dtype('float64')): _pressio_io_data_from_numpy_2d_double,
  (3, numpy.dtype('float64')): _pressio_io_data_from_numpy_3d_double,
  (4, numpy.dtype('float64')): _pressio_io_data_from_numpy_4d_double,
  (1, numpy.dtype('uint8')): _pressio_io_data_from_numpy_1d_uint8_t,
  (2, numpy.dtype('uint8')): _pressio_io_data_from_numpy_2d_uint8_t,
  (3, numpy.dtype('uint8')): _pressio_io_data_from_numpy_3d_uint8_t,
  (4, numpy.dtype('uint8')): _pressio_io_data_from_numpy_4d_uint8_t,
  (1, numpy.dtype('int8')): _pressio_io_data_from_numpy_1d_int8_t,
  (2, numpy.dtype('int8')): _pressio_io_data_from_numpy_2d_int8_t,
  (3, numpy.dtype('int8')): _pressio_io_data_from_numpy_3d_int8_t,
  (4, numpy.dtype('int8')): _pressio_io_data_from_numpy_4d_int8_t,
  (1, numpy.dtype('uint16')): _pressio_io_data_from_numpy_1d_uint16_t,
  (2, numpy.dtype('uint16')): _pressio_io_data_from_numpy_2d_uint16_t,
  (3, numpy.dtype('uint16')): _pressio_io_data_from_numpy_3d_uint16_t,
  (4, numpy.dtype('uint16')): _pressio_io_data_from_numpy_4d_uint16_t,
  (1, numpy.dtype('int16')): _pressio_io_data_from_numpy_1d_int16_t,
  (2, numpy.dtype('int16')): _pressio_io_data_from_numpy_2d_int16_t,
  (3, numpy.dtype('int16')): _pressio_io_data_from_numpy_3d_int16_t,
  (4, numpy.dtype('int16')): _pressio_io_data_from_numpy_4d_int16_t,
  (1, numpy.dtype('uint32')): _pressio_io_data_from_numpy_1d_uint32_t,
  (2, numpy.dtype('uint32')): _pressio_io_data_from_numpy_2d_uint32_t,
  (3, numpy.dtype('uint32')): _pressio_io_data_from_numpy_3d_uint32_t,
  (4, numpy.dtype('uint32')): _pressio_io_data_from_numpy_4d_uint32_t,
  (1, numpy.dtype('int32')): _pressio_io_data_from_numpy_1d_int32_t,
  (2, numpy.dtype('int32')): _pressio_io_data_from_numpy_2d_int32_t,
  (3, numpy.dtype('int32')): _pressio_io_data_from_numpy_3d_int32_t,
  (4, numpy.dtype('int32')): _pressio_io_data_from_numpy_4d_int32_t,
  (1, numpy.dtype('uint64')): _pressio_io_data_from_numpy_1d_uint64_t,
  (2, numpy.dtype('uint64')): _pressio_io_data_from_numpy_2d_uint64_t,
  (3, numpy.dtype('uint64')): _pressio_io_data_from_numpy_3d_uint64_t,
  (4, numpy.dtype('uint64')): _pressio_io_data_from_numpy_4d_uint64_t,
  (1, numpy.dtype('int64')): _pressio_io_data_from_numpy_1d_int64_t,
  (2, numpy.dtype('int64')): _pressio_io_data_from_numpy_2d_int64_t,
  (3, numpy.dtype('int64')): _pressio_io_data_from_numpy_3d_int64_t,
  (4, numpy.dtype('int64')): _pressio_io_data_from_numpy_4d_int64_t,
}
__pressio_to_numpy = {
  (1, _pressio.float_dtype) : _pressio_io_data_to_numpy_1d_float,
  (2, _pressio.float_dtype) : _pressio_io_data_to_numpy_2d_float,
  (3, _pressio.float_dtype) : _pressio_io_data_to_numpy_3d_float,
  (4, _pressio.float_dtype) : _pressio_io_data_to_numpy_4d_float,
  (1, _pressio.double_dtype) : _pressio_io_data_to_numpy_1d_double,
  (2, _pressio.double_dtype) : _pressio_io_data_to_numpy_2d_double,
  (3, _pressio.double_dtype) : _pressio_io_data_to_numpy_3d_double,
  (4, _pressio.double_dtype) : _pressio_io_data_to_numpy_4d_double,
  (1, _pressio.int8_dtype) : _pressio_io_data_to_numpy_1d_int8_t,
  (2, _pressio.int8_dtype) : _pressio_io_data_to_numpy_2d_int8_t,
  (3, _pressio.int8_dtype) : _pressio_io_data_to_numpy_3d_int8_t,
  (4, _pressio.int8_dtype) : _pressio_io_data_to_numpy_4d_int8_t,
  (1, _pressio.int16_dtype) : _pressio_io_data_to_numpy_1d_int16_t,
  (2, _pressio.int16_dtype) : _pressio_io_data_to_numpy_2d_int16_t,
  (3, _pressio.int16_dtype) : _pressio_io_data_to_numpy_3d_int16_t,
  (4, _pressio.int16_dtype) : _pressio_io_data_to_numpy_4d_int16_t,
  (1, _pressio.int32_dtype) : _pressio_io_data_to_numpy_1d_int32_t,
  (2, _pressio.int32_dtype) : _pressio_io_data_to_numpy_2d_int32_t,
  (3, _pressio.int32_dtype) : _pressio_io_data_to_numpy_3d_int32_t,
  (4, _pressio.int32_dtype) : _pressio_io_data_to_numpy_4d_int32_t,
  (1, _pressio.int64_dtype) : _pressio_io_data_to_numpy_1d_int64_t,
  (2, _pressio.int64_dtype) : _pressio_io_data_to_numpy_2d_int64_t,
  (3, _pressio.int64_dtype) : _pressio_io_data_to_numpy_3d_int64_t,
  (4, _pressio.int64_dtype) : _pressio_io_data_to_numpy_4d_int64_t,
  (1, _pressio.uint8_dtype) : _pressio_io_data_to_numpy_1d_uint8_t,
  (2, _pressio.uint8_dtype) : _pressio_io_data_to_numpy_2d_uint8_t,
  (3, _pressio.uint8_dtype) : _pressio_io_data_to_numpy_3d_uint8_t,
  (4, _pressio.uint8_dtype) : _pressio_io_data_to_numpy_4d_uint8_t,
  (1, _pressio.uint16_dtype) : _pressio_io_data_to_numpy_1d_uint16_t,
  (2, _pressio.uint16_dtype) : _pressio_io_data_to_numpy_2d_uint16_t,
  (3, _pressio.uint16_dtype) : _pressio_io_data_to_numpy_3d_uint16_t,
  (4, _pressio.uint16_dtype) : _pressio_io_data_to_numpy_4d_uint16_t,
  (1, _pressio.uint32_dtype) : _pressio_io_data_to_numpy_1d_uint32_t,
  (2, _pressio.uint32_dtype) : _pressio_io_data_to_numpy_2d_uint32_t,
  (3, _pressio.uint32_dtype) : _pressio_io_data_to_numpy_3d_uint32_t,
  (4, _pressio.uint32_dtype) : _pressio_io_data_to_numpy_4d_uint32_t,
  (1, _pressio.uint64_dtype) : _pressio_io_data_to_numpy_1d_uint64_t,
  (2, _pressio.uint64_dtype) : _pressio_io_data_to_numpy_2d_uint64_t,
  (3, _pressio.uint64_dtype) : _pressio_io_data_to_numpy_3d_uint64_t,
  (4, _pressio.uint64_dtype) : _pressio_io_data_to_numpy_4d_uint64_t,
}
__pressio_to_np_dtype = {
  _pressio.float_dtype : numpy.float32,
}
__pressio_to_np_dtype = {
  _pressio.float_dtype : numpy.float32,
  _pressio.double_dtype : numpy.double,
  _pressio.uint8_dtype : numpy.uint8,
  _pressio.int8_dtype : numpy.int8,
  _pressio.uint16_dtype : numpy.uint16,
  _pressio.int16_dtype : numpy.int16,
  _pressio.uint32_dtype : numpy.uint32,
  _pressio.int32_dtype : numpy.int32,
  _pressio.uint64_dtype : numpy.uint64,
  _pressio.int64_dtype : numpy.int64,
}


def io_data_from_numpy(array):
  length = len(array.shape)
  dtype = array.dtype
  return __pressio_from_numpy[length, dtype](array)

def io_data_to_numpy(ptr):
  num_dims = data_num_dimensions(ptr)
  dtype = data_dtype(ptr)
  return __pressio_to_numpy[num_dims, dtype](ptr)

%}

%feature("autodoc", 3);
%include "pressio.h"
%include "pressio_compressor.h"
%ignore pressio_data_new_nonowning;
%ignore pressio_data_new_owning;
%ignore pressio_data_new_move;
%ignore pressio_data_new_copy;
%ignore pressio_data_new_empty;
%include "pressio_data.h"
%include "pressio_dtype.h"
%include "pressio_metrics.h"
%include "pressio_option.h"
%include "pressio_options.h"
%include "pressio_options_iter.h"
%include "libpressio_ext/io/pressio_io.h"
%include "libpressio_ext/io/posix.h"
