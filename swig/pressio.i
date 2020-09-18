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

%define pressio_numpy_type(type)
  %apply (type* INPLACE_ARRAY1, int DIM1 ) {( type * data, size_t r1)};
  %apply (type* INPLACE_ARRAY2, int DIM1, int DIM2 ) { ( type * data, size_t r1, size_t r2)};
  %apply (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3 ) {( type* data, size_t r1, size_t r2, size_t r3)};
  %apply (type* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 ) {( type* data, size_t r1, size_t r2, size_t r3, size_t r4)};
namespace std {
  %template( vector_ ## type ) vector< type >;
}
  %template( _pressio_io_data_to_numpy_ ## type ) _pressio_io_data_to_numpy< type >;
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
  _pressio.float_dtype : _pressio_io_data_to_numpy_float,
  _pressio.double_dtype : _pressio_io_data_to_numpy_double,
  _pressio.uint8_dtype : _pressio_io_data_to_numpy_uint8_t,
  _pressio.int8_dtype : _pressio_io_data_to_numpy_int8_t,
  _pressio.uint16_dtype : _pressio_io_data_to_numpy_uint16_t,
  _pressio.int16_dtype : _pressio_io_data_to_numpy_int16_t,
  _pressio.uint32_dtype : _pressio_io_data_to_numpy_uint32_t,
  _pressio.int32_dtype : _pressio_io_data_to_numpy_int32_t,
  _pressio.uint64_dtype : _pressio_io_data_to_numpy_uint64_t,
  _pressio.int64_dtype : _pressio_io_data_to_numpy_int64_t,
}

def io_data_from_numpy(array):
  length = len(array.shape)
  dtype = array.dtype
  return __pressio_from_numpy[length, dtype](array)

def io_data_to_numpy(ptr):
  num_dims = data_num_dimensions(ptr)
  dtype = data_dtype(ptr)
  dims = [data_get_dimension(ptr, i) for i in range(num_dims)]
  return numpy.reshape(__pressio_to_numpy[dtype](ptr), dims)


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
