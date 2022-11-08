#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include <string>

#include "pressio_posix.h"
#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

namespace libpressio { namespace cufile_io_ns {

struct pressio_cufile_metadata {
    CUfileHandle_t cf_handle;
    int32_t close_driver;
    int fd;
};

extern "C" void cufile_free_fn(void* data, void* metadata){
  auto cf_metadata = reinterpret_cast<pressio_cufile_metadata* >(metadata);
  cudaFree(data);
  cuFileHandleDeregister(cf_metadata->cf_handle);
  if(cf_metadata->close_driver) {
    cuFileDriverClose();
  }
  close(cf_metadata->fd);
  delete cf_metadata;
}


class cufile_plugin : public libpressio_io_plugin {
  public:
  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
    if(buf == nullptr) {
      set_error(1, "metadata is required");
      return nullptr;
    }

    //open the file
    int fd = open(path.c_str(), O_RDONLY|O_DIRECT);
    if (fd < 0) {
      set_error(errno, errno_to_error());
      return nullptr;
    }

    //optionally open the driver
    CUfileError_t status;
    if(open_close_driver) {
      status = cuFileDriverOpen();
      if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        set_error(status.err, cufileop_status_error(status.err));
        return nullptr;
      }
    }
    CUfileDescr_t cf_descr;
    memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));

    //register the file handle
    CUfileHandle_t cf_handle;
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if(status.err != CU_FILE_SUCCESS) {
      if (open_close_driver) {cuFileDriverClose();}
      close(fd);
      set_error(status.err, cufileop_status_error(status.err));
      return nullptr;
    }

    //allocate memory for the buffer
    void* dev_ptr;
    cudaError_t cuda_result = cudaMalloc(&dev_ptr, buf->size_in_bytes());
    if(cuda_result != static_cast<int>(CUDA_SUCCESS)) {
      cuFileHandleDeregister(cf_handle);
      if(open_close_driver) { cuFileDriverClose(); }
      close(fd);
      set_error(static_cast<int>(cuda_result), cudaGetErrorString(cuda_result));
      return nullptr;
    }

    //register the buffer
    status = cuFileBufRegister(dev_ptr, buf->size_in_bytes(), buf_register_flags);
    if (status.err != CU_FILE_SUCCESS) {
      cudaFree(dev_ptr);
      cuFileHandleDeregister(cf_handle);
      if(open_close_driver) { cuFileDriverClose(); }
      close(fd);
      set_error(static_cast<int>(cuda_result), cudaGetErrorString(cuda_result));
    }

    //preform the actual read
    ssize_t read = cuFileRead(cf_handle, dev_ptr, buf->size_in_bytes(), static_cast<off_t>(file_offset), static_cast<off_t>(dev_offset));
    if(read == -1) {
      //posix error
      set_error(errno, errno_to_error());
      cudaFree(dev_ptr);
      cuFileHandleDeregister(cf_handle);
      if(open_close_driver) { cuFileDriverClose(); }
      close(fd);
      return nullptr;
    } else if (read < 0) {
      //cuFileError
      set_error(static_cast<int>(-read), cufileop_status_error(static_cast<CUfileOpError>(-read)));
      cudaFree(dev_ptr);
      cuFileHandleDeregister(cf_handle);
      if(open_close_driver) { cuFileDriverClose(); }
      close(fd);
      return nullptr;
    } else if (static_cast<size_t>(read) < buf->size_in_bytes()) {
      //short read
      set_error(1, "short read from file");
      cudaFree(dev_ptr);
      cuFileHandleDeregister(cf_handle);
      if(open_close_driver) { cuFileDriverClose(); }
      close(fd);
      return nullptr;
    } else {
      auto metadata = new pressio_cufile_metadata{
        cf_handle,
        open_close_driver,
        fd
      };
      return new pressio_data(pressio_data::move(
          buf->dtype(),
          dev_ptr,
          buf->dimensions(),
          cufile_free_fn,
          metadata
          ));
    }



    set_error(2, "pressio cufile_read: unexpected error");
    return nullptr;
  }
  virtual int write_impl(struct pressio_data const*) override{
    return set_error(1, "not supported");
  }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "io:path", path);
    set(options, "cufile:open_close_driver", open_close_driver);
    set(options, "cufile:buf_register_flags", buf_register_flags);
    set(options, "cufile:file_offset", file_offset);
    set(options, "cufile:dev_offset", dev_offset);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "io:path", &path);
    get(options, "cufile:open_close_driver", &open_close_driver);
    get(options, "cufile:buf_register_flags", &buf_register_flags);
    get(options, "cufile:file_offset", &file_offset);
    get(options, "cufile:dev_offset", &dev_offset);
    return 0;
  }

  const char* version() const override { return "0.0.1"; }
  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "experimental");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "io plugin to load data using NVIDIA's cufile APIs");
    set(opt, "io:path", "path to read the file from");
    set(opt, "cufile:open_close_driver", "should cuFileDriverOpen and cuFileDriverClose be called?");
    set(opt, "cufile:buf_register_flags", "flags passed to cuFileBufRegister");
    set(opt, "cufile:file_offset", "offset to use when reading from the file");
    set(opt, "cufile:dev_offset", "offset to use when reading/writing to device memory");
    return opt;
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<cufile_plugin>(*this);
  }
  const char* prefix() const override {
    return "cufile";
  }

  private:
  int32_t open_close_driver = 1;
  std::string path;
  int32_t buf_register_flags = 0;
  int64_t file_offset = 0;
  int64_t dev_offset = 0;
};

static pressio_register io_cufile_plugin(io_plugins(), "cufile", [](){ return compat::make_unique<cufile_plugin>(); });
}}
