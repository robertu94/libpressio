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


%include "pressio_version.h"
%include "pybuffer.i"
%pybuffer_string(const char* compressor_id)
%pybuffer_binary(const char* buffer, size_t buffer_size)


%include <std_string.i>
%include <std_vector.i>
%include <cpointer.i>

%pointer_functions(bool, bool)
%pointer_functions(int8_t, int8)
%pointer_functions(uint8_t, uint8)
%pointer_functions(int16_t, int16)
%pointer_functions(uint16_t, uint16)
%pointer_functions(int32_t, int32)
%pointer_functions(uint32_t, uint32)
%pointer_functions(int64_t, int64)
%pointer_functions(uint64_t, uint64)
%pointer_functions(double, double)
%pointer_functions(float, float)

#if LIBPRESSIO_HAS_MPI4PY
%include "mpi4py/mpi4py.i"
%mpi4py_typemap(Comm, MPI_Comm)
#endif
%include "pypressio.h"

%numpy_typemaps(bool       , NPY_BOOL     , size_t)
%numpy_typemaps(signed char       , NPY_BYTE     , size_t)
%numpy_typemaps(unsigned char     , NPY_UBYTE    , size_t)
%numpy_typemaps(short             , NPY_SHORT    , size_t)
%numpy_typemaps(unsigned short    , NPY_USHORT   , size_t)
%numpy_typemaps(int               , NPY_INT      , size_t)
%numpy_typemaps(unsigned int      , NPY_UINT     , size_t)
%numpy_typemaps(long              , NPY_LONG     , size_t)
%numpy_typemaps(unsigned long     , NPY_ULONG    , size_t)
%numpy_typemaps(long long         , NPY_LONGLONG , size_t)
%numpy_typemaps(unsigned long long, NPY_ULONGLONG, size_t)
%numpy_typemaps(float             , NPY_FLOAT    , size_t)
%numpy_typemaps(double            , NPY_DOUBLE   , size_t)
%numpy_typemaps(std::int8_t            , NPY_INT8     , size_t)
%numpy_typemaps(std::int16_t           , NPY_INT16    , size_t)
%numpy_typemaps(std::int32_t           , NPY_INT32    , size_t)
%numpy_typemaps(std::int64_t           , NPY_INT64    , size_t)
%numpy_typemaps(std::uint8_t           , NPY_UINT8    , size_t)
%numpy_typemaps(std::uint16_t          , NPY_UINT16   , size_t)
%numpy_typemaps(std::uint32_t          , NPY_UINT32   , size_t)
%numpy_typemaps(std::uint64_t          , NPY_UINT64   , size_t)
%numpy_typemaps(bool       , NPY_BOOL     , long int)
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
%numpy_typemaps(signed char       , NPY_BYTE     , long int)
%numpy_typemaps(unsigned char     , NPY_UBYTE    , long int)
%numpy_typemaps(short             , NPY_SHORT    , long int)
%numpy_typemaps(unsigned short    , NPY_USHORT   , long int)
%numpy_typemaps(int               , NPY_INT      , long int)
%numpy_typemaps(unsigned int      , NPY_UINT     , long int)
%numpy_typemaps(long              , NPY_LONG     , long int)
%numpy_typemaps(unsigned long     , NPY_ULONG    , long int)
%numpy_typemaps(long long         , NPY_LONGLONG , long int)
%numpy_typemaps(unsigned long long, NPY_ULONGLONG, long int)

%define pressio_numpy_type(type, name)

  %apply (type* INPLACE_ARRAY1, size_t DIM1 ) {( type * data, size_t r1)};
  %apply (type* INPLACE_ARRAY2, size_t DIM1, size_t DIM2 ) { ( type * data, size_t r1, size_t r2)};
  %apply (type* INPLACE_ARRAY3, size_t DIM1, size_t DIM2, size_t DIM3 ) {( type* data, size_t r1, size_t r2, size_t r3)};
  %apply (type* INPLACE_ARRAY4, size_t DIM1, size_t DIM2, size_t DIM3, size_t DIM4 ) {( type* data, size_t r1, size_t r2, size_t r3, size_t r4)};
  %apply (type** ARGOUTVIEWM_ARRAY1, long int* DIM1) {( type** ptr_argout, long int* r1)};
  %apply (type** ARGOUTVIEWM_ARRAY2, long int* DIM1, long int* DIM2) {( type** ptr_argout, long int* r1, long int* r2)};
  %apply (type** ARGOUTVIEWM_ARRAY3, long int* DIM1, long int* DIM2, long int* DIM3) {( type** ptr_argout, long int* r1, long int* r2, long int* r3)};
  %apply (type** ARGOUTVIEWM_ARRAY4, long int* DIM1, long int* DIM2, long int* DIM3, long int* DIM4) {( type** ptr_argout, long int* r1, long int* r2, long int* r3, long int* r4)};
namespace std {
  %template( vector_ ## name ) vector< type >;
}
  %template( _pressio_io_data_to_numpy_1d_ ## name ) _pressio_io_data_to_numpy_1d< type >;
  %template( _pressio_io_data_to_numpy_2d_ ## name ) _pressio_io_data_to_numpy_2d< type >;
  %template( _pressio_io_data_to_numpy_3d_ ## name ) _pressio_io_data_to_numpy_3d< type >;
  %template( _pressio_io_data_to_numpy_4d_ ## name ) _pressio_io_data_to_numpy_4d< type >;
  %template( _pressio_io_data_from_numpy_1d_ ## name ) _pressio_io_data_from_numpy_1d< type >;
  %template( _pressio_io_data_from_numpy_2d_ ## name ) _pressio_io_data_from_numpy_2d< type >;
  %template( _pressio_io_data_from_numpy_3d_ ## name ) _pressio_io_data_from_numpy_3d< type >;
  %template( _pressio_io_data_from_numpy_4d_ ## name ) _pressio_io_data_from_numpy_4d< type >;
%enddef

pressio_numpy_type(bool, bool);
pressio_numpy_type(float, float);
pressio_numpy_type(double, double);
pressio_numpy_type(unsigned char, uint8_t);
pressio_numpy_type(unsigned short, uint16_t);
pressio_numpy_type(unsigned int, uint32_t);
pressio_numpy_type(unsigned long int, uint64_t);
pressio_numpy_type(signed char, int8_t);
pressio_numpy_type(short, int16_t);
pressio_numpy_type(int, int32_t);
pressio_numpy_type(long int, int64_t);

namespace std { 
  %template() vector<size_t>;
  %template(vector_string) vector<std::string>;
}

%rename("%(strip:[pressio_])s") "";

%pythoncode %{
import numpy
__pressio_from_numpy = {
  (1, numpy.dtype('bool')): _pressio_io_data_from_numpy_1d_bool,
  (2, numpy.dtype('bool')): _pressio_io_data_from_numpy_2d_bool,
  (3, numpy.dtype('bool')): _pressio_io_data_from_numpy_3d_bool,
  (4, numpy.dtype('bool')): _pressio_io_data_from_numpy_4d_bool,
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
  (1, _pressio.byte_dtype) : _pressio_io_data_to_numpy_1d_uint8_t,
  (2, _pressio.byte_dtype) : _pressio_io_data_to_numpy_2d_uint8_t,
  (3, _pressio.byte_dtype) : _pressio_io_data_to_numpy_3d_uint8_t,
  (4, _pressio.byte_dtype) : _pressio_io_data_to_numpy_4d_uint8_t,
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
  (1, _pressio.bool_dtype) : _pressio_io_data_to_numpy_1d_bool,
  (2, _pressio.bool_dtype) : _pressio_io_data_to_numpy_2d_bool,
  (3, _pressio.bool_dtype) : _pressio_io_data_to_numpy_3d_bool,
  (4, _pressio.bool_dtype) : _pressio_io_data_to_numpy_4d_bool,
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
  _pressio.bool_dtype : bool,
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
%ignore pressio_option_new_strings;
%include "pressio_option.h"
%include "pressio_options.h"
%include "pressio_options_iter.h"
%include "libpressio_ext/io/pressio_io.h"
%include "libpressio_ext/io/posix.h"
