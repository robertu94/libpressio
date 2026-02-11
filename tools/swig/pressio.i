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
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/io/pressio_io.h"
#include "libpressio_ext/io/posix.h"
#include "pypressio.h"
#include "libpressio_ext/highlevel/libpressio_highlevel.h"
#if LIBPRESSIO_HAS_JSON
#include "libpressio_ext/json/pressio_options_json.h"
#endif
#if LIBPRESSIO_HAS_OPENSSL
#include "libpressio_ext/hash/libpressio_hash.h"
#endif
%}

%include <stdint.i>


%include "pressio_version.h"

#if SWIGPYTHON
%include "pybuffer.i"
%pybuffer_string(const char* compressor_id)
%pybuffer_binary(const char* buffer, size_t buffer_size)
#endif


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

#if SWIGPYTHON
#if LIBPRESSIO_HAS_MPI4PY
%include "mpi4py/mpi4py.i"
%mpi4py_typemap(Comm, MPI_Comm)
%newobject options_new_comm;
#endif
#endif


%include "pypressio.h"

%define pressio_numpy_type(type, name)
namespace std {
  %template( vector_ ## name ) vector< type >;
}
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
  %template(vector_data) vector<struct pressio_data*>;
  %template(vector_string) vector<std::string>;
}

%rename("%(strip:[pressio_])s") "";


%ignore pressio_new_metrics;
%newobject pressio_get_compressor;
%newobject pressio_new_metric;
%delobject pressio_release;
//defined in pypressio.h instead
%newobject new_metrics;
%include "pressio.h"
%newobject pressio_compressor_get_documentation;
%newobject pressio_compressor_get_configuration;
%newobject pressio_compressor_get_options;
%newobject pressio_compressor_get_metrics_results;
%newobject pressio_compressor_get_metrics;
%newobject pressio_compressor_clone;
%delobject pressio_compressor_release;
%include "pressio_compressor.h"
//prefer the versions using std::vector from pypressio.h
%ignore pressio_data_new_nonowning;
%ignore pressio_data_new_owning;
%ignore pressio_data_new_move;
%ignore pressio_data_new_copy;
%ignore pressio_data_new_empty;
//these are defined in pypressio.h
%newobject data_new_copy;
%newobject data_new_nowning;
%newobject data_new_empty;
%newobject data_new_move;


%delobject pressio_data_free;
%include "pressio_data.h"
%include "pressio_dtype.h"
%newobject pressio_metrics_get_results;
%newobject pressio_metrics_get_options;
%newobject pressio_metrics_get_documentation;
%newobject pressio_metrics_get_configuration;
%newobject pressio_metrics_clone;
%newobject pressio_metrics_evaluate;
%delobject pressio_metrics_free;
%include "pressio_metrics.h"
%ignore pressio_option_new_strings;
%ignore pressio_option_get_strings;
%newobject pressio_option_new_integer8;
%newobject pressio_option_new_integer16;
%newobject pressio_option_new_integer;
%newobject pressio_option_new_integer64;
%newobject pressio_option_new_uinteger8;
%newobject pressio_option_new_uinteger16;
%newobject pressio_option_new_uinteger;
%newobject pressio_option_new_uinteger64;
%newobject pressio_option_new_float;
%newobject pressio_option_new_double;
%newobject pressio_option_new_bool;
%newobject pressio_option_new_dtype;
%newobject pressio_option_new_threadsafety;
%newobject pressio_option_new_data;
%newobject pressio_option_new_userptr;
%newobject pressio_option_new_userptr_managed;
%newobject pressio_option_new;
%newobject pressio_option_get_data;
// these are defined in pypressio.h
%newobject option_new_string;
%newobject option_new_strings;
%delobject pressio_option_free;
%include "pressio_option.h"
%newobject pressio_options_get_iter;
%newobject pressio_options_new;
%delobject pressio_options_free;

%include "pressio_options.h"
%newobject pressio_options_iter_get_value;
%delobject pressio_options_iter_free;
%include "pressio_options_iter.h"
%newobject pressio_get_io;
%newobject pressio_io_get_configuration;
%newobject pressio_io_get_documentation;
%newobject pressio_io_get_options;
%newobject pressio_io_read;
%newobject pressio_io_clone;
%delobject pressio_io_free;
%include "libpressio_ext/io/pressio_io.h"
%newobject pressio_io_data_fread;
%newobject pressio_io_data_read;
%newobject pressio_io_data_path_read;
%include "libpressio_ext/io/posix.h"


#if LIBPRESSIO_HAS_JSON
%newobject pressio_options_new_json;
%newobject pressio_options_to_json;
%include "libpressio_ext/json/pressio_options_json.h"
#endif
#if LIBPRESSIO_HAS_OPENSSL
%newobject pressio_options_hashkeys;
%newobject pressio_options_hashentries;
%include "libpressio_ext/hash/libpressio_hash.h"
#endif
%newobject pressio_highlevel_get_compressor;
%newobject pressio_highlevel_get_io;
%include "libpressio_ext/highlevel/libpressio_highlevel.h"

%include  "dlpack/dlpack.h"

%pythoncode %{
    class PressioData:
        def __init__(self, ptr):
            self.ptr = ptr
        def __del__(self):
            data_free(self.ptr)
        def __dlpack__(self, stream = None, max_version = None, dl_device = None, copy = None):
            if max_version is None:
                # Keep and use the DLPack 0.X implementation
                return 
            else:
                max_version_swig = optional_max_version() if max_version is None else optional_max_version(max_version[0], max_version[1])
                dl_device_swig = optional_dl_device() if dl_device is None else optional_dl_device(dl_device[0], dl_device[1])
                copy_swig = optional_copy() if copy is None else optional_copy(copy)

                # We get to produce `DLManagedTensorVersioned` now.
                if max_version >= (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION):
                    # Consumer understands us, just return a Capsule with our max version
                    return data_to_dlpack_versioned(self.ptr, max_version_swig, dl_device_swig, copy_swig)
                elif max_version[0] == DLPACK_MAJOR_VERSION:
                    # major versions match, we should still be fine here -
                    # return our own max version
                    return data_to_dlpack_versioned(self.ptr, max_version_swig, dl_device_swig, copy_swig)
                else:
                    # if we're at a higher major version internally, did we
                    # keep an implementation of the older major version around?
                    # For example, if the producer is on DLPack 1.x and the consumer
                    # is 0.y, can the producer still export a capsule containing
                    # DLManagedTensor and not DLManagedTensorVersioned?
                    # If so, use that. Else, the producer should raise a BufferError
                    # here to tell users that the consumer's max_version is too
                    # old to allow the data exchange to happen.
                    return data_to_dlpack_managed(self.ptr)
%}
