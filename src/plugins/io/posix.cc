#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <errno.h>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"
#include "pressio_posix.h"





extern "C" {
  struct pressio_data* pressio_io_data_read(struct pressio_data* dims, int in_filedes) {
    pressio_data* ret;
    if(dims != nullptr) {
      if(pressio_data_has_data(dims)) {
        //re-use the buffer provided by dims
        ret = new pressio_data(std::move(*dims));
      } else {
        //create a new buffer of the appropriate size
        auto dtype = pressio_data_dtype(dims);
        auto dims_v = dims->dimensions();
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
               size_t total_read = 0;
               size_t bytes_read = 0;
               while((bytes_read = read(in_filedes, ((uint8_t*)pressio_data_ptr(ret, nullptr))+total_read, pressio_data_get_bytes(ret) - total_read)) > 0) {
                               total_read += bytes_read;
               }
    if(total_read != pressio_data_get_bytes(ret)) {
      pressio_data_free(ret);
      return nullptr;
    } else  {
      return ret;
    }
  }

  struct pressio_data* pressio_io_data_fread(struct pressio_data* dims, FILE* in_file) {
    if(in_file == nullptr) return nullptr;
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


  size_t pressio_io_data_fwrite(struct pressio_data const* data, FILE* out_file) {

    return pressio_io_data_write(data, fileno(out_file));
  }

  size_t pressio_io_data_write(struct pressio_data const* data, int out_filedes) {
    size_t total_written = 0;
    size_t bytes_written = 0;
    while((bytes_written = write(out_filedes, ((uint8_t*)pressio_data_ptr(data, nullptr)) + total_written, pressio_data_get_bytes(data) - total_written)) > 0) {
      total_written += bytes_written;
    }
    return total_written;
  }

  size_t pressio_io_data_path_write(struct pressio_data const* data, const char* path) {
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

namespace libpressio { namespace io { namespace posix_ns {
struct posix_io : public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data* data) override {
    errno = 0;
    if(path) {
        auto ret = pressio_io_data_path_read(data, path->c_str());
        if(ret == nullptr) {
          if(errno != 0)set_error(2, errno_to_error());
          else set_error(3, "invalid dims");
        }
        return ret;
    }
    if(file_ptr) {
      auto ret = pressio_io_data_fread(data, *file_ptr);
      if(ret == nullptr) {
        if(errno != 0)set_error(2, errno_to_error());
        else set_error(3, "invalid dims");
      }
      return ret;
    }
    if(fd) {
      auto ret = pressio_io_data_read(data, *fd);
      if(ret == nullptr) {
        if(errno != 0) set_error(2, errno_to_error());
        else set_error(3, "invalid dims");
      }
      return ret;
    }

    invalid_configuration();
    return nullptr;
  }

  virtual int write_impl(struct pressio_data const* indata) override{
    auto data = domain_manager().make_readable(domain_plugins().build("malloc"), *indata);
    errno = 0;
    if(path) {
      int ret = pressio_io_data_path_write(&data, path->c_str()) != data.size_in_bytes();
      if(ret) {
        if(errno) return set_error(2, errno_to_error());
        else return set_error(3, "unknown failure");
      }
      return ret;
    }
    else if(file_ptr) {
      int ret = pressio_io_data_fwrite(&data, *file_ptr) != data.size_in_bytes();
      if(ret) {
        if(errno) set_error(2, errno_to_error());
        else set_error(3, "unknown failure");
      }
      return ret;
    }
    else if(fd) {
      int ret = pressio_io_data_write(&data, *fd) != data.size_in_bytes();
      if(ret) {
        if(errno) set_error(2, errno_to_error());
        else set_error(3, "unknown failure");
      }
      return ret;
    }
    return invalid_configuration();
  }
  virtual struct pressio_options get_configuration_impl() const override{
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe",  pressio_thread_safety_single);
    return opts;
  }

  virtual int set_options_impl(struct pressio_options const& options) override{
    std::string path;
    if(get(options, "io:path", &path) == pressio_options_key_set) {
      this->path = path;
    } else {
      this->path = {};
    }

    void* file_ptr;
    if(get(options, "io:file_pointer", &file_ptr ) == pressio_options_key_set) {
      this->file_ptr = (FILE*)file_ptr;
    } else {
      this->file_ptr = {};
    }

    int fd;
    if(get(options, "io:file_descriptor", &fd) == pressio_options_key_set) {
      this->fd = fd;
    } else {
      this->fd = {};
    }
    return 0;
  }
  struct pressio_options get_documentation_impl() const override{
    pressio_options opts;
    set(opts, "pressio:description", "POSIX io");
    set(opts, "io:path", "path on the file system to read/write from");
    set(opts, "io:file_pointer", "FILE* to read/write from");
    set(opts, "io:file_descriptor", "posix file descriptor to read/write from");
    return opts;
  }
  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;

    if(path) set(opts, "io:path", *path);
    else set_type(opts, "io:path", pressio_option_charptr_type);

    if(file_ptr) set(opts, "io:file_pointer", (void*)*file_ptr);
    else set_type(opts, "io:file_pointer", pressio_option_userptr_type);

    if(fd) set(opts, "io:file_descriptor", *fd);
    else set_type(opts, "io:file_descriptor", pressio_option_int32_type);

    return opts;
  }

  int patch_version() const override{ 
    return 1;
  }
  virtual const char* version() const override{
    return "0.0.1";
  }
  const char* prefix() const override {
    return "posix";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<posix_io>(*this);
  }

  private:
  int invalid_configuration() {
    return set_error(1, "invalid configuration");
  }

  compat::optional<std::string> path;
  compat::optional<FILE*> file_ptr;
  compat::optional<int> fd;
};

pressio_register registration(io_plugins(), "posix", [](){ return compat::make_unique<posix_io>(); });
} } }
