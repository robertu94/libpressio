#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "pressio.h"
#include "libpressio_ext/cpp/compressor.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include "pressio_version.h"
#include "dlpack/dlpack.h"

#if LIBPRESSIO_HAS_MPI4PY
#include <mpi.h>


void options_set_comm(struct pressio_options* options, const char* key, MPI_Comm comm);

pressio_option* option_new_comm(MPI_Comm comm);

#endif

struct numpy_array_interface {
    std::vector<size_t> shape;
    std::string typestr;
    bool read_only;
    intptr_t ptr;
    int version;
};

numpy_array_interface data_to_array_interface(pressio_data* data); 

/**
 * \returns the DLDevice the data resides on
 */
DLDevice data_to_dlpack_device(pressio_data* data);
/**
 * \returns the deprecated dlpack DLManagedTensor for Backwards compatability
 */
DLManagedTensor data_to_dlpack_managed(pressio_data* data);

/*
 * swig doesn't have support for std::optional, when it does optional_max_verison, optional_dl_device, and optional_copy could be replaced
 */

struct optional_max_version {
    optional_max_version(): has_version(false), major(0), minor(0) {}
    optional_max_version(int major, int minor): has_version(true), major(major), minor(minor) {}
    operator bool() const { return has_version; }
    bool has_version;
    int major;
    int minor;
};

struct optional_dl_device {
    optional_dl_device(): has_device(false), type(0), index(0) {}
    optional_dl_device(int type, int index): has_device(true), type(type), index(index) {}
    operator bool() const { return has_device; }
    bool has_device;
    int type;
    int index;
};

struct optional_copy {
    optional_copy(): has_copy(false), copy(false) {}
    optional_copy(bool copy): has_copy(true), copy(true) {}
    operator bool() const { return has_copy; }
    bool has_copy;
    bool copy;
};

/**
 * \returns the current dlpack versioned tensor
 */
DLManagedTensorVersioned data_to_dlpack_versioned(pressio_data* data, optional_max_version max_version, optional_dl_device dl_device, optional_copy copy);

std::vector<std::string> option_get_strings(pressio_option const* options) ;

void option_set_strings(pressio_option* options, std::vector<std::string> const& strings) ;

std::vector<uint64_t> data_dimensions(const pressio_data* data) ;
intptr_t data_ptr(const pressio_data* data) ;

struct pressio_option* option_new_strings(std::vector<std::string> const& strings) ;
struct pressio_option* option_new_string(std::string const& string) ;

struct pressio_data* data_new_empty(const pressio_dtype dtype, std::vector<uint64_t> dimensions) ;
struct pressio_data* data_new_nonowning(const pressio_dtype dtype, void* data, std::vector<uint64_t> dimensions) ;
struct pressio_data* data_new_nonowning_ptr(const pressio_dtype dtype, intptr_t data, std::vector<uint64_t> dimensions) ;
struct pressio_data* data_new_copy(const enum pressio_dtype dtype, void* src, std::vector<uint64_t>  dimensions) ;
struct pressio_data* data_new_owning(const pressio_dtype dtype, std::vector<uint64_t> dimensions) ;
struct pressio_data* data_new_move(const pressio_dtype dtype,
    void* data,
    std::vector<uint64_t> dimensions,
    pressio_data_delete_fn deleter,
    void* metadata) ;

pressio_metrics* new_metrics(struct pressio* library, std::vector<std::string> metrics) ;

int compressor_compress_many(struct pressio_compressor* compressor, std::vector<struct pressio_data*> const& inputs, std::vector<struct pressio_data*>& outputs) ;
int compressor_decompress_many(struct pressio_compressor* compressor, std::vector<struct pressio_data*> const& inputs, std::vector<struct pressio_data*>& outputs) ;

void options_set_strings(pressio_options* options, std::string const& key, std::vector<std::string> const& values);
