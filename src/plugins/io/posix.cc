#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "pressio_data.h"
#include "libpressio_ext/io/posix.h"

namespace {
  std::vector<size_t> get_all_dimentions(struct pressio_data const* data) {
    std::vector<size_t> dims;
    if(data) {
      for (size_t i = 0; i < pressio_data_num_dimentions(data); ++i) {
        dims.emplace_back(pressio_data_get_dimention(data, i));
      }
    }
    return dims;
  }
}

extern "C" {

  struct pressio_data* pressio_io_data_read(struct pressio_data* dims, int in_filedes) {
    pressio_data* ret;
    if(dims != nullptr) {
      if(pressio_data_has_data(dims)) {
        //re-use the buffer provided by dims
        ret = dims;
      } else {
        //create a new buffer of the appropriate size
        auto dtype = pressio_data_dtype(dims);
        auto dims_v = get_all_dimentions(dims);
        pressio_data_free(dims);
        ret = pressio_data_new_owning(dtype, dims_v.size(), dims_v.data());
      }
    } else {
      struct stat statbuf;
      if(fstat(in_filedes, &statbuf)) {
        return nullptr;
      } 
      size_t size = static_cast<size_t>(statbuf.st_size); 
      ret = pressio_data_new_owning(pressio_byte_dtype, 1, &size);
    }
    read(in_filedes, pressio_data_ptr(ret, nullptr), pressio_data_get_bytes(ret));
    return ret;
  }

  struct pressio_data* pressio_io_data_fread(struct pressio_data* dims, FILE* in_file) {
    return pressio_io_data_read(dims, fileno(in_file));
  }


  struct pressio_data* pressio_io_data_path_read(struct pressio_data* dims, const char* path) {
    FILE* in_file = fopen(path, "r");
    if(in_file != nullptr) {
      auto ret = pressio_io_data_fread(dims, in_file);
      fclose(in_file);
      return ret;
    } else {
      return nullptr;
    }
  }


  size_t pressio_io_data_fwrite(struct pressio_data* data, FILE* out_file) {

    return fwrite(pressio_data_ptr(data, nullptr),
        pressio_dtype_size(pressio_data_dtype(data)),
        pressio_data_num_elements(data),
        out_file
        );
  }

  size_t pressio_io_data_write(struct pressio_data* data, int out_filedes) {
    return write(out_filedes,
        pressio_data_ptr(data, nullptr),
        pressio_data_get_bytes(data)
        );
  }

  size_t pressio_io_path_path_write(struct pressio_data* data, const char* path) {
    FILE* out_file = fopen(path, "w");
    if(out_file != nullptr) {
      auto ret = pressio_io_data_fwrite(data, out_file);
      fclose(out_file);
      return ret;
    } else {
      return 0;
    }
  }
  
}
