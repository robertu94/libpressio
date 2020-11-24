#include <cmath> //for exp and log
#include <iterator>
#include <map>
#include <sstream>
#include <bg/bg.h>
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

#define INVALID_TYPE -1

namespace {
  struct bg_iless {
    bool operator()(std::string lhs, std::string rhs) const {
      std::transform(std::begin(lhs), std::end(lhs), std::begin(lhs), [](unsigned char c){return std::tolower(c);});
      std::transform(std::begin(rhs), std::end(rhs), std::begin(rhs), [](unsigned char c){return std::tolower(c);});
      return lhs < rhs;
    }
  };
  static std::map<std::string, int, bg_iless> const bitgroom_mode_str_to_code {
    {"bitgroom", BITGROOM},
    {"bitshave", BITSHAVE},
    {"bitset", BITSET},
  };
  static std::map<std::string, int, bg_iless> const bitgroom_ec_mode_str_to_code {
    {"nsd", BG_NSD},
    {"dsd", BG_DSD},
  };

  template <class Map>
  std::vector<std::string> bg_keys(Map const& map) {
    std::vector<std::string> keys;
    keys.reserve(map.size());
    std::transform(
        std::begin(map), std::end(map), std::back_inserter(keys),
        [](typename Map::const_reference it) { return it.first; });
    return keys;
  }
}


class bit_grooming_plugin: public libpressio_compressor_plugin {
  public:
    bit_grooming_plugin() {
      std::stringstream ss;
      ss << bit_grooming_plugin::major_version() << "." << bit_grooming_plugin::minor_version() << "." << bit_grooming_plugin::patch_version() << "." << bit_grooming_plugin::revision_version();
      bg_version = ss.str();
    };
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;

      set(options, "bit_grooming:mode", bgMode);
      set_type(options, "bit_grooming:mode_str", pressio_option_charptr_type);

      set(options, "bit_grooming:error_control_mode", errorControlMode);
      set_type(options, "bit_grooming:error_control_mode_str", pressio_option_charptr_type);

      set(options, "bit_grooming:n_sig_digits", nun_sig_digits); // number of significant digits
      set(options, "bit_grooming:n_sig_decimals", num_sig_decimals); // number of significant decimal digits
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set(options, "pressio:thread_safe", static_cast<int>(pressio_thread_safety_multiple));
      set(options, "bit_grooming:mode", bg_keys(bitgroom_mode_str_to_code));
      set(options, "bit_grooming:error_control_mode_mode", bg_keys(bitgroom_ec_mode_str_to_code));
      return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get(options, "bit_grooming:mode", &bgMode);

      std::string tmp_mode;
      if(get(options, "bit_grooming:mode_str", &tmp_mode) == pressio_options_key_set) {
        auto const& it = bitgroom_mode_str_to_code.find(tmp_mode);
        if(it != bitgroom_mode_str_to_code.end()) {
          bgMode = it->second;
        }
      }
      if(get(options, "bit_grooming:error_control_mode_str", &tmp_mode) == pressio_options_key_set) {
        auto const& it = bitgroom_ec_mode_str_to_code.find(tmp_mode);
        if(it != bitgroom_ec_mode_str_to_code.end()) {
          errorControlMode = it->second;
        }
      }
      get(options, "bit_grooming:error_control_mode", &errorControlMode);
      get(options, "bit_grooming:n_sig_digits", &nun_sig_digits); // number of significant digits
      get(options, "bit_grooming:n_sig_decimals",
          &num_sig_decimals); // number of significant decimal digits
      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_bg_type(pressio_data_dtype(input));
      if(type == INVALID_TYPE) {
         return INVALID_TYPE;
      }

      size_t nbEle = pressio_data_num_elements(input);
      unsigned long outSize;
      void* data = pressio_data_ptr(input, nullptr);
      unsigned char* compressed_data;

      compressed_data = BG_compress_args(type, data, &outSize, bgMode, errorControlMode,
                                         nun_sig_digits, num_sig_decimals, nbEle);

      if(compressed_data == NULL)
      {
        return set_error(2, "Error when bit grooming is compressing the data");
      }

      *output = pressio_data::move(pressio_byte_dtype, compressed_data, 1, &outSize, pressio_data_libc_free_fn, nullptr);
      return 0;
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      int type = libpressio_type_to_bg_type(pressio_data_dtype(output));
      if(type == INVALID_TYPE) {
         return INVALID_TYPE;
      }
      unsigned char* bytes = (unsigned char*)pressio_data_ptr(input, nullptr);
      size_t nbEle = pressio_data_num_elements(output);
      size_t byteLength = pressio_data_get_bytes(input);

      void* decompressed_data = BG_decompress(type, bytes, byteLength, nbEle);
      *output = pressio_data::move(pressio_data_dtype(output), decompressed_data, 1, &nbEle, pressio_data_libc_free_fn, nullptr);
      return 0;
      }


    
    int major_version() const override {
      return BG_VER_MAJOR;
    }
    int minor_version() const override {
      return BG_VER_MINOR;
    }
    int patch_version() const override {
      return BG_VER_BUILD;
    }
    int revision_version () const { 
      return BG_VER_REVISION;
    }

    const char* version() const override {
      return bg_version.c_str(); 
    }


    const char* prefix() const override {
      return "bit_grooming";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<bit_grooming_plugin>(*this);
    }
  private:
    std::string bg_version;
    int libpressio_type_to_bg_type(pressio_dtype type)
    {
      if(type == pressio_float_dtype)
      {
        return BG_FLOAT;
      }
      else if(type == pressio_double_dtype)
      {
        return BG_DOUBLE;
      }
      else
      {
        set_error(2, "Invalid data type");
        return INVALID_TYPE;
      }
    }

    int bgMode = BITGROOM;
    int errorControlMode = BG_NSD;
    int nun_sig_digits = 5;
    int num_sig_decimals = 5;
};

static pressio_register compressor_bit_grooming_plugin(compressor_plugins(), "bit_grooming", [](){return compat::make_unique<bit_grooming_plugin>(); });
