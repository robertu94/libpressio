#include <H5Dpublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <H5Ppublic.h>
#include <H5Fpublic.h>
#include <H5public.h>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <pressio_data.h>
#include <cassert>
#include <vector>
#include <string>
#include <pressio_compressor.h>
#include <std_compat/std_compat.h>
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "std_compat/memory.h"

namespace {
  compat::optional<pressio_dtype> h5t_to_pressio(hid_t h5type) {
    if(H5Tequal(h5type, H5T_NATIVE_INT8) > 0) return pressio_int8_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_INT16) > 0) return pressio_int16_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_INT32) > 0) return pressio_int32_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_INT64) > 0) return pressio_int64_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_UINT8) > 0) return pressio_uint8_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_UINT16) > 0) return pressio_uint16_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_UINT32) > 0) return pressio_uint32_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_UINT64) > 0) return pressio_uint64_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_FLOAT) > 0) return pressio_float_dtype;
    if(H5Tequal(h5type, H5T_NATIVE_DOUBLE) > 0) return pressio_double_dtype;
    return compat::optional<pressio_dtype>{};
  }
  hid_t pressio_to_h5t(pressio_dtype dtype) {
    switch(dtype) {
      case pressio_double_dtype: 
        return H5T_NATIVE_DOUBLE;
      case pressio_float_dtype:
        return H5T_NATIVE_FLOAT;
      case pressio_uint8_dtype:
        return H5T_NATIVE_UINT8;
      case pressio_uint16_dtype:
        return H5T_NATIVE_UINT16;
      case pressio_uint32_dtype:
        return H5T_NATIVE_UINT32;
      case pressio_uint64_dtype:
        return H5T_NATIVE_UINT64;
      case pressio_int8_dtype:
        return H5T_NATIVE_INT8;
      case pressio_int16_dtype:
        return H5T_NATIVE_INT16;
      case pressio_int32_dtype:
        return H5T_NATIVE_INT32;
      case pressio_int64_dtype:
        return H5T_NATIVE_INT64;
      case pressio_byte_dtype:
        return H5T_NATIVE_UCHAR;
      default:
        assert(false && "unexpected type");
        //shutup gcc
        return H5T_NATIVE_UCHAR;
    }
  }


  bool hdf_path_exists(hid_t file, std::string const& path) {
    if(path == std::string("") or path == std::string("/")) return true;
    else {
      //check for parent path
      auto last_slash_pos = path.find_last_of('/');
      if(last_slash_pos == 0) {
        return H5Lexists(file, path.c_str(), H5P_DEFAULT);
      }
      else if(last_slash_pos != std::string::npos)
      {
        //recurse to check for parent
        auto parent = path.substr(0, last_slash_pos);
        if (not hdf_path_exists(file, parent)) return false;
      } 

      //check the path passed in
      return H5Lexists(file, path.c_str(), H5P_DEFAULT);
    }
  }

  /**
   * this class is a standard c++ idiom for closing resources
   * it calls the function passed in during the destructor.
   */
  template <class Function>
  class cleanup {
    public:
      cleanup(Function f) noexcept: cleanup_fn(std::move(f)), do_cleanup(true) {}
      cleanup(cleanup&& rhs) noexcept: cleanup_fn(std::move(rhs.cleanup_fn)), do_cleanup(true) {
        do_cleanup = false;
      }
      cleanup(cleanup const&)=delete;
      cleanup& operator=(cleanup const&)=delete;
      cleanup& operator=(cleanup && rhs) noexcept { 
        do_cleanup = compat::exchange(rhs.do_cleanup, false);
        cleanup_fn = std::move(rhs.cleanup_fn);
      }
      ~cleanup() { if(do_cleanup) cleanup_fn(); }

    private:
      Function cleanup_fn;
      bool do_cleanup;
  };
  template<class Function>
  auto make_cleanup(Function&& f) -> decltype(cleanup<Function>(std::declval<Function>())) {
    return cleanup<Function>(std::forward<Function>(f));
  }
}

extern "C" {

struct pressio_data*
pressio_io_data_path_h5read(const char* file_name, const char* dataset_name)
{
  pressio instance;
  auto hdf_io = instance.get_io("hdf5");
  hdf_io->set_options({
      {"io:path", std::string(file_name)},
      {"hdf5:dataset", std::string(dataset_name)},
      });

  return hdf_io->read(nullptr);
}

int
pressio_io_data_path_h5write(struct pressio_data const* data, const char* file_name, const char* dataset_name)
{
  pressio instance;
  auto hdf_io = instance.get_io("hdf5");
  hdf_io->set_options({
      {"io:path", std::string(file_name)},
      {"hdf5:dataset", std::string(dataset_name)},
      });
  return hdf_io->write(data);
}

}

struct hdf5_io: public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data* buffer) override {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file < 0) {
      set_error(1, "failed to open file " + filename);
      return nullptr;
    }
    auto cleanup_file = make_cleanup([&]{ H5Fclose(file); });

    hid_t dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);
    if(dataset < 0) {
      set_error(2, "failed to open dataset" + dataset_name);
      return nullptr;
    }
    auto cleanup_dataset = make_cleanup([&]{ H5Dclose(dataset); });

    hid_t filespace = H5Dget_space(dataset);
    if(filespace < 0) {
      set_error(3, "failed to get dataspace from file");
      return nullptr;
    }
    auto cleanup_dataspace = make_cleanup([&]{ H5Sclose(filespace); });
    int ndims = H5Sget_simple_extent_ndims(filespace);
    std::vector<hsize_t> read_file_extent(ndims);
    H5Sget_simple_extent_dims(filespace, read_file_extent.data(), nullptr);


    if(should_prepare_read()) {
      if(not prepare_filespace(filespace)) {
        set_error(6, "invalid hyperslab selection");
        return nullptr;
      }
    }

    //convert to size_t from hsize_t
    std::vector<size_t> pressio_size(ndims);
    switch(H5Sget_select_type(filespace)) {
      case H5S_SEL_HYPERSLABS:
        if(H5Sis_regular_hyperslab(filespace) > 0) {
          std::vector<hsize_t> start(ndims), count(ndims), stride(ndims), block(ndims);
          H5Sget_regular_hyperslab(filespace,
              start.data(), stride.data(), count.data(), block.data());
          for (int i = 0; i < ndims; ++i) {
            pressio_size[i] = count[i] * block[i];
          }
        } else {
          //TODO support non regular hyperslabs
          set_error(7, "non-regular hyperslabs are not supported");
          return nullptr;
        }
        break;
      case H5S_SEL_ALL:
        //when the selection is all points, use the extents
        H5Sget_simple_extent_dims(filespace, read_file_extent.data(), nullptr);
        std::copy(std::begin(read_file_extent), std::end(read_file_extent), std::begin(pressio_size));
        break;
      default:
        set_error(8, "other selection types are not supported");
        return nullptr;

    }

    hid_t type = H5Dget_type(dataset);
    if(type < 0) {
      set_error(4, "failed to get datatype");
      return nullptr;
    }
    auto cleanup_type = make_cleanup([&]{ H5Tclose(type);}) ;

    {
      auto dtype = h5t_to_pressio(type);
      if(dtype) {
        pressio_data* ret;
        if(buffer == nullptr) {
          ret = pressio_data_new_owning(*dtype, pressio_size.size(), pressio_size.data());
        } else {
          ret = buffer;
        }
        auto ptr = pressio_data_ptr(ret, nullptr);
        std::vector<hsize_t> memextents(std::begin(pressio_size), std::end(pressio_size));
        hid_t memspace = H5Screate_simple(ndims,  memextents.data(), nullptr);
        if(memspace < 0) {
          set_error(9, "failed to create memspace");
          return nullptr;
        }
        auto memspace_cleanup = make_cleanup([&]{ H5Sclose(memspace); });

        if(H5Dread(dataset, type, memspace, filespace, H5P_DEFAULT, ptr) < 0) {
          set_error(10, "read failed");
          return nullptr;
        }

        return ret;
      } else {
        set_error(5, "unknown datatype");
        return nullptr;
      }
    }
  }

  virtual int write_impl(struct pressio_data const* data) override{
    //check if the file exists
    hid_t file;
    int perms_ok = access(filename.c_str(), W_OK);
    if(perms_ok == 0)
    {
      file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    } else {
      if(errno == ENOENT) {
        file = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
      } else {
        char err_msg[1024];
        std::fill(err_msg, err_msg+1024, '\0');
        strerror_r(errno, err_msg, 1024);
        set_error(1, err_msg);
        return 1;
      }
    }
    if(file < 0) {
      set_error(2, "failed to open file" + filename);
      return 1;
    }
    auto cleanup_file = make_cleanup([&]{ H5Fclose(file); });



    hid_t dataset;
    if (hdf_path_exists(file, dataset_name))
    {
      dataset = H5Dopen(file, dataset_name.c_str(), H5P_DEFAULT);
    } else {
      //create a filespace to create the dataset
      std::vector<hsize_t> h5_dims(data->num_dimensions());
      if(file_extent.empty()) {
        std::vector<size_t> dims = data->dimensions();
        std::copy(std::begin(dims), std::end(dims), std::begin(h5_dims));
      } else {
        std::copy(std::begin(file_extent), std::end(file_extent), std::begin(h5_dims));
      }
      hid_t creation_filespace = H5Screate_simple(
          h5_dims.size(),
          h5_dims.data(),
          nullptr
          );
      if(creation_filespace < 0) {
        set_error(3, "failed to create dataspace");
        return 1;
      }
      auto cleanup_memspace = make_cleanup([&]{ H5Sclose(creation_filespace);});
      dataset = H5Dcreate2(file,
          dataset_name.c_str(),
          pressio_to_h5t(pressio_data_dtype(data)),
          creation_filespace,
          H5P_DEFAULT,
          H5P_DEFAULT,
          H5P_DEFAULT
          );
    }
    if(dataset < 0) {
      set_error(4, "failed to create or open dataset " + dataset_name);
      return 1;
    }
    auto cleanup_dataset = make_cleanup([&]{ H5Dclose(dataset);});

    //create the memspace

    //create the filespace
    hid_t filespace = H5Dget_space(dataset);
    if(filespace < 0) {
      return set_error(5, "failed to get file dataspace");
    }
    auto cleanup_filespace = make_cleanup([&]{H5Sclose(filespace);});
    if(should_prepare_write()) {
      if(not prepare_filespace(filespace)) {
        return set_error(6, "invalid hyperslab selection");
      }
    }

    //prepare the memspace
    std::vector<size_t> pressio_dims = data->dimensions();
    std::vector<hsize_t> memspace_dims(std::begin(pressio_dims), std::end(pressio_dims));
    hid_t memspace = H5Screate_simple(static_cast<int>(memspace_dims.size()), memspace_dims.data(), nullptr);
    if(memspace < 0) {
      return set_error(7, "failed to create memspace");
    }
    auto cleanup_memspace = make_cleanup([&]{ H5Sclose(memspace); });

    //write the dataset
    return (H5Dwrite(
        dataset,
        pressio_to_h5t(pressio_data_dtype(data)),
        memspace,
        filespace,
        H5P_DEFAULT,
        pressio_data_ptr(data,nullptr)
        ) < 0);
  }

  virtual struct pressio_options get_configuration_impl() const override{
    return {
      {"pressio:thread_safe",  static_cast<int32_t>(pressio_thread_safety_single)}
    };
  }

  virtual int set_options_impl(struct pressio_options const& options) override{
    get(options, "io:path", &filename);
    get(options, "hdf5:dataset", &dataset_name);
    pressio_data tmp;
    if(get(options, "hdf5:file_start", &tmp) == pressio_options_key_set) {
      auto file_start_t = tmp.to_vector<uint64_t>();
      file_start.assign(std::begin(file_start_t), std::end(file_start_t));
    }
    if(get(options, "hdf5:file_count", &tmp) == pressio_options_key_set) {
      auto file_count_t = tmp.to_vector<uint64_t>();
      file_count.assign(std::begin(file_count_t), std::end(file_count_t));
    }
    if(get(options, "hdf5:file_stride", &tmp) == pressio_options_key_set) {
      auto file_stride_t = tmp.to_vector<uint64_t>();
      file_stride.assign(std::begin(file_stride_t), std::end(file_stride_t));
    }
    if(get(options, "hdf5:file_block", &tmp) == pressio_options_key_set) {
      auto file_block_t = tmp.to_vector<uint64_t>();
      file_block.assign(std::begin(file_block_t), std::end(file_block_t));
    }
    if(get(options, "hdf5:file_extent", &tmp) == pressio_options_key_set) {
      auto file_extent_t = tmp.to_vector<uint64_t>();
      file_extent.assign(std::begin(file_extent_t), std::end(file_extent_t));
    }
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;
    set(opts, "io:path", filename);
    set(opts, "hdf5:dataset", dataset_name);
    auto to_uint64v = [](std::vector<hsize_t> const& hsv) {
      std::vector<uint64_t> ui64v(std::begin(hsv), std::end(hsv));
      return pressio_data(std::begin(ui64v), std::end(ui64v));
    };
    set(opts, "hdf5:file_block", to_uint64v(file_block));
    set(opts, "hdf5:file_count", to_uint64v(file_count));
    set(opts, "hdf5:file_stride", to_uint64v(file_stride));
    set(opts, "hdf5:file_start", to_uint64v(file_stride));
    set(opts, "hdf5:file_extent", to_uint64v(file_extent));
    return opts;
  }

  int patch_version() const override{ 
    return 1;
  }
  virtual const char* version() const override{
    return "0.0.1";
  }

  const char* prefix() const override {
    return "hdf5";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<hdf5_io>(*this);
  }

  private:

  bool should_prepare_read() const {
    if(file_start.empty() && file_block.empty() && file_count.empty() && file_stride.empty()) return false;
    else return true;
  }
  bool should_prepare_write() const {
    if(file_start.empty() && file_block.empty() && file_count.empty() && file_stride.empty()) return false;
    else return true;
  }
  bool prepare_filespace(hid_t filespace) const {
      const int ndims = H5Sget_simple_extent_ndims(filespace);

      std::vector<hsize_t> read_start;
      if(file_start.empty()) {
        read_start = std::vector<hsize_t>(ndims, 0);
      } else {
        read_start = file_start;
      }

      std::vector<hsize_t> read_count;
      if(file_count.empty()) {
        read_count = std::vector<hsize_t>(ndims, 1);
      } else {
        read_count = file_count;
      }

      hsize_t const* read_block = nullptr;
      if(!file_block.empty()) {
        read_block = file_block.data();
      }

      hsize_t const* read_stride = nullptr;
      if(!file_stride.empty()) {
        read_stride = file_stride.data();
      }

      H5Sselect_hyperslab(
          filespace,
          H5S_SELECT_SET,
          read_start.data(),
          read_stride,
          read_count.data(),
          read_block
          );

      if(H5Sselect_valid(filespace) <= 0) {
        return false;
      }
      return true;
  }

  std::string filename;
  std::string dataset_name;
  std::vector<hsize_t> file_block, file_start, file_count, file_stride, file_extent;
};

static pressio_register io_hdf5_plugin(io_plugins(), "hdf5", [](){ return compat::make_unique<hdf5_io>(); });
