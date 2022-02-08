#include <vector>
#include <regex>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "std_compat/memory.h"
#include "std_compat/bit.h"
#include "std_compat/string_view.h"

namespace libpressio { namespace numpy {
const size_t libpressio_numpy_max_v1_length = ~uint16_t{0};

std::string libpressio_read_data(std::string const& path) {
  std::ifstream infile(path, std::ios::binary);
  if(not (infile.is_open() && infile.good())) {
    throw std::runtime_error("failed to read " + path);
  }
  struct stat buf = {};
  stat(path.c_str(), &buf);
  const size_t size = buf.st_size;

  std::string data(size, 0);
  infile.read(&data[0], size);

  return data;
}

struct libpressio_npy_data {
  compat::string_view magic;
  unsigned int major_version;
  unsigned int minor_version;
  compat::string_view header;
  pressio_dtype type;
  std::vector<size_t> dims;
  bool fortran_order = false;
};

int libpressio_write_np(std::ostream& out, struct pressio_data const* data) {
  auto const& dtype = data->dtype();
  auto dims = data->dimensions();
  std::reverse(dims.begin(), dims.end()); //numpy expects C ordered dimensions
  
  std::stringstream shape;
  if(dims.size() == 0) {
    throw std::runtime_error("empty pressio_data not supported");
  } else if(dims.size() == 1) {
    shape << '(' << dims.front() << ",)";
  } else {
    shape << '(' << dims[0];
    for (size_t i = 1; i < dims.size(); ++i) {
      shape << ", " << dims[i];
    }
    shape << "), ";
  }

  std::stringstream format;
  if(compat::endian::native == compat::endian::big){
    format << '>';
  } else {
    format << '<';
  }
  switch(dtype) {
    case pressio_bool_dtype:
      format << "b1";
      break;
    case pressio_int8_dtype:
      format << "i1";
      break;
    case pressio_int16_dtype:
      format << "i2";
      break;
    case pressio_int32_dtype:
      format << "i4";
      break;
    case pressio_int64_dtype:
      format << "i8";
      break;
    case pressio_byte_dtype:
    case pressio_uint8_dtype:
      format << "u1";
      break;
    case pressio_uint16_dtype:
      format << "u2";
      break;
    case pressio_uint32_dtype:
      format << "u4";
      break;
    case pressio_uint64_dtype:
      format << "u8";
      break;
    case pressio_float_dtype:
      format << "f4";
      break;
    case pressio_double_dtype:
      format << "f8";
      break;
  }

  std::stringstream header;
  header << '{'
    << "'descr': '" << format.rdbuf() << "', "
    << "'fortran_order': False, " 
    << "'shape': " << shape.rdbuf()
    << "}\n";
  std::string header_str = header.str();
  const auto metadata_len = 
    1 + // major_version
    1 + // minor_version
    ((header_str.size() + 10 > libpressio_numpy_max_v1_length)? 4: 2) + //header_len (v1)
    6 // magiclen
    ;
  const auto current_length = header.str().size() + metadata_len;
  const auto padding_len = ((current_length % 16 == 0)? 0: (16-(current_length % 16)));
  for (size_t i = 0; i < padding_len; ++i) {
    header << '\x20';
  }

  uint8_t major_version = 1, minor_version = 0;
  if(header_str.size() > libpressio_numpy_max_v1_length) {
    major_version = 2;
  }

  out << "\x93NUMPY";
  out.write(reinterpret_cast<char const*>(&major_version), 1);
  out.write(reinterpret_cast<char const*>(&minor_version), 1);
  if(header_str.size() > libpressio_numpy_max_v1_length) {
    uint32_t header_len = header.str().size();
    out.write(reinterpret_cast<char const*>(&header_len), 4);
  } else {
    uint16_t header_len = header.str().size();
    out.write(reinterpret_cast<char const*>(&header_len), 2);
  }
  
  out << header.rdbuf();
  out.write(reinterpret_cast<char const*>(data->data()), data->size_in_bytes());

  return 0;
}

int parse_header(std::string const& header, libpressio_npy_data& np) {
  const std::regex header_fmt(
    "^\\{\\s*"
      "'descr':\\s+'([^']+)',\\s+"
      "'fortran_order':\\s+((?:False)|(?:True)),\\s+"
      "'shape':\\s+\\(((?:\\d+,?\\s*)+)\\)"
    "[^}]*"
    "\\}"
  );
  std::smatch match;
  if(std::regex_search(header, match, header_fmt)) {
    np.type = [](std::string const& type_str) {
      std::regex layout_fmt(
          "([<>|!@=])" //layout
          "([xcbBhHiIlLqQnNfduU])" //type
          "(\\d*)" //size in bytes
      );
      std::smatch type_match;
      if(std::regex_match(type_str, type_match, layout_fmt)) {
        char byte_order = *type_match[1].first;
        switch(byte_order) {
          case '<': //little
            if(compat::endian::native != compat::endian::little) throw std::runtime_error("unsupported endian-ness");
            break;
          case '>': //big-endian
            if(compat::endian::native != compat::endian::big) throw std::runtime_error("unsupported endian-ness");
            break;
          case '|': //not-relevant, ok
          case '=': //native, ok
          case '@': //native, ok
            break;
        }

        char type_value = *type_match[2].first;
        size_t size = std::stoull(std::string(type_match[3].first, type_match[3].second));
        switch (type_value) {
        case 'f':
          if(size == 4) {
            return pressio_float_dtype;
          } else if(size == 8) {
            return pressio_double_dtype;
          }
          break;
        case 'i':
          if(size == 1) {
            return pressio_int8_dtype;
          } else if(size == 2) {
            return pressio_int16_dtype;
          } else if(size == 4) {
            return pressio_int32_dtype;
          } else if(size == 8) {
            return pressio_int64_dtype;
          }
          break;
        case 'u':
          if(size == 1) {
            return pressio_uint8_dtype;
          } else if(size == 2) {
            return pressio_uint16_dtype;
          } else if(size == 4) {
            return pressio_uint32_dtype;
          } else if(size == 8) {
            return pressio_uint64_dtype;
          }
          break;
        case 'b':
          return pressio_bool_dtype;
        default:
          break;
        }
      } else {
        throw std::runtime_error("unsupported format " + type_str);
      }
      throw std::runtime_error("unsupported format " + type_str);
    }(std::string(match[1].first, match[1].second));
    auto const order_str = std::string(match[2].first, match[2].second);
    if(order_str == "True") {
      np.fortran_order = true;
    } else if (order_str == "False") {
      np.fortran_order = false;
    } else {
      throw std::runtime_error("unexpected order");
    }
    np.dims = [](std::string const& dim_str){
      std::vector<size_t> d;
      std::regex digits("\\d+");
      auto begin = std::sregex_iterator(dim_str.begin(), dim_str.end(), digits);
      auto end = std::sregex_iterator();
      std::transform(begin, end, std::back_inserter(d), [](
            typename decltype(begin)::value_type const& match
            ) {
            return std::stoull(std::string(match[0].first, match[0].second));
          });

      return d;
    }(std::string(match[3].first, match[3].second));
  }
  return 0;
}

struct numpy_io : public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
    try{
      auto data  = libpressio_read_data(path);
      libpressio_npy_data np;
      size_t current = 0;
      char* ptr = &data[0];
      np.magic = compat::string_view(ptr, current+=6);
      np.major_version = *reinterpret_cast<const uint8_t*>(ptr + current++);
      np.minor_version = *reinterpret_cast<const uint8_t*>(ptr + current++);
      uint32_t header_len;
      if(np.major_version == 1) {
        header_len = *reinterpret_cast<const uint16_t*>(ptr + (current+=2)-2);
      } else if (np.major_version == 2) {
        header_len = *reinterpret_cast<const uint32_t*>(ptr + (current+=4)-4);
      } else {
        throw std::runtime_error("unsupported major_version");
      }
      np.header = compat::string_view(ptr+10, header_len-1);
      current += header_len;
      parse_header(std::string(np.header), np);

      std::reverse(np.dims.begin(), np.dims.end()); //numpy provides C ordered dimensions
      if(buf && buf->dimensions() == np.dims && buf->dtype() == np.type) {
        pressio_data* out = pressio_data_new_empty(pressio_byte_dtype, 0, nullptr);
        *out = std::move(*buf);
        auto mem = out->data();
        std::memcpy(
            mem,
            ptr+current,
            out->size_in_bytes()
            );
        return out;
      } else {
        return pressio_data_new_copy(
          np.type,
          ptr+current,
          np.dims.size(),
          np.dims.data()
          );
      }
    } catch(std::exception const& ex) {
      set_error(1, ex.what());
      return nullptr;
    }
  }

  virtual int write_impl(struct pressio_data const* data) override{
    std::ofstream ofs(path, std::ios::binary| std::ios::trunc | std::ios::out);
    return libpressio_write_np(ofs, data);
  }

  virtual struct pressio_options get_configuration_impl() const override{
    pressio_options opts;
    set(opts, "pressio:thread_safe",  static_cast<int32_t>(pressio_thread_safety_single));
    set(opts, "pressio:stability", "stable");
    return opts;
  }

  virtual int set_options_impl(struct pressio_options const& options) override{
    get(options, "io:path", &path);
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;
    set(opts, "io:path", path);
    return opts;
  }
  virtual struct pressio_options get_documentation_impl() const override{
    pressio_options opts;
    set(opts, "pressio:description", "read Numpy .npy files");
    set(opts, "io:path", "path to the file on disk");
    return opts;
  }


  int patch_version() const override{ 
    return 1;
  }
  virtual const char* version() const override{
    return "0.0.1";
  }
  const char* prefix() const override {
    return "numpy";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<numpy_io>(*this);
  }

  private:
  std::string path;
};

static pressio_register io_posix_plugin(io_plugins(), "numpy", [](){ return compat::make_unique<numpy_io>(); });
} }
