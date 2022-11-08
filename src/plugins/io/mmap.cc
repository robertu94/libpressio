#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
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
#include "std_compat/memory.h"
#include "pressio_posix.h"

extern "C" {
  struct pressio_mmap_metadata{
    size_t size;
    int fd;
    int close_file;
  };

  void pressio_data_libc_unmmap(void* data, void* metadata_ptr) {
    assert(data != nullptr && "data cannot be nullptr for pressio_data_libc_unmmap");
    assert(metadata_ptr != nullptr && "metadata cannot nullptr for pressio_data_libc_unmmap");

    auto metadata = reinterpret_cast<pressio_mmap_metadata*>(metadata_ptr);
    munmap(data, metadata->size);
    if(metadata->close_file) {
      close(metadata->fd);
    }
    delete metadata;
  }
}

namespace libpressio { namespace mmap_plugin {

struct mmap_io : public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data* data) override {
    errno = 0;
    if(path) {
        return io_data_path_read(data, path->c_str());
    }
    if(fd) {
      return io_data_read(data, *fd);
    }

    invalid_configuration();
    return nullptr;
  }

  virtual int write_impl(struct pressio_data const*) override{
    return set_error(1, "not implemented");
  }
  virtual struct pressio_options get_configuration_impl() const override{
    pressio_options opts;
    set(opts, "pressio:thread_safe",  pressio_thread_safety_single);
    set(opts, "pressio:stability", "stable");
    return opts;
  }

  virtual int set_options_impl(struct pressio_options const& options) override{
    std::string path;
    if(get(options, "io:path", &path) == pressio_options_key_set) {
      this->path = path;
    } else {
      this->path = {};
    }
    int fd;
    if(get(options, "io:file_descriptor", &fd) == pressio_options_key_set) {
      this->fd = fd;
    } else {
      this->fd = {};
    }
    return 0;
  }
  virtual struct pressio_options get_documentation_impl() const override{
    pressio_options opts;
    set(opts, "pressio:description", "uses mmap shared mappings to read files");
    set(opts, "io:path", "path to the file on disk");
    set(opts, "io:file_descriptor", "file descriptor for the file on disk");
    return opts;
  }

  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;

    if(path) set(opts, "io:path", *path);
    else set_type(opts, "io:path", pressio_option_charptr_type);

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
    return "mmap";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<mmap_io>(*this);
  }

  private:
  int invalid_configuration() {
    return set_error(1, "invalid configuration");
  }

  size_t determine_size(pressio_data const* data, int fd) {
    struct stat statbuf = {};
    if(fstat(fd, &statbuf) == -1) {
      set_error(4, errno_to_error());
    }
    const size_t expected_size_in_bytes = (data)? data->size_in_bytes() : 0;
    const size_t real_size_in_bytes = static_cast<size_t>(statbuf.st_size);
    if(data == nullptr || real_size_in_bytes != expected_size_in_bytes) {
      if(data != nullptr) {
        set_error(2, "unexpected size");
      }
      return real_size_in_bytes;
    } else {
      return expected_size_in_bytes;
    }

  }

  //we are opening our own FD, we need to close it
  pressio_data* io_data_path_read(pressio_data* data, const char* path) {
    auto metadata = std::make_unique<pressio_mmap_metadata>();
    metadata->close_file = true;
    metadata->fd = open(path, O_RDONLY);
    if(metadata->fd == -1) {
      set_error(5, errno_to_error() + path);
      return nullptr;
    }
    metadata->size = determine_size(data, metadata->fd);
    return io_data_read_common(data, std::move(metadata));
  }

  //we are opening our own FD, we need to close it
  pressio_data* io_data_read(pressio_data* data, int fd) {
    auto metadata = std::make_unique<pressio_mmap_metadata>();
    metadata->close_file = false;
    metadata->size = determine_size(data, fd);
    metadata->fd = fd;
    return io_data_read_common(data, std::move(metadata));
  }
  pressio_data* io_data_read_common(pressio_data* data, std::unique_ptr<pressio_mmap_metadata>&& metadata) {
    if(error_code()) {
      return nullptr;
    }
    void* addr = mmap(nullptr, metadata->size, PROT_READ, MAP_SHARED, metadata->fd, 0);
    if(addr == MAP_FAILED) {
      set_error(3, "mapping failed");
      return nullptr;
    }

    return new pressio_data(pressio_data::move(
        (data)? data->dtype() : pressio_byte_dtype,
        addr,
        (data)? data->dimensions() : std::vector<size_t>{metadata->size},
        pressio_data_libc_unmmap,
        metadata.release()
        ));
  }

  compat::optional<std::string> path;
  compat::optional<int> fd;
};

static pressio_register io_mmap_plugin(io_plugins(), "mmap", [](){ return compat::make_unique<mmap_io>(); });

} } 
