#include "std_compat/memory.h"
#include <std_compat/string_view.h>
#include <std_compat/bit.h>
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include <nlohmann/json.hpp>
#include "libpressio_ext/cpp/json.h"
#include <sstream>

namespace libpressio { namespace manifest_ns {


class manifest_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "manifest:compressor", impl_id, impl);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(get_threadsafe(*impl)));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "manifest:compressor", R"(compressor to use to decompress)", impl);
    set(options, "pressio:description", R"(compressor plugin that records meta data from compression to enable reconstruction)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "manifest:compressor", compressor_plugins(), impl_id, impl);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    nlohmann::json j{};
    j["t"] = input->dtype(); //dtype
    j["d"] = input->dimensions(); //dims
    j["o"] = get_options(); //options
    j["n"] = get_name(); //name
    j["c"] = versions(); //versions tree
    std::vector<uint8_t> msgpk = nlohmann::json::to_msgpack(j);

    int rc = impl->compress(input, output);
    if(rc) {
      return set_error(impl->error_code(), impl->error_msg());
    }
    auto tmp = pressio_data::owning(
        pressio_byte_dtype,
        { sizeof(uint32_t) +
          sizeof(uint64_t) +
          msgpk.size() +
          output->size_in_bytes()}
        );
    uint32_t version = 1;
    uint64_t header_size = msgpk.size();
    if (compat::endian::native == compat::endian::big) {
      version = compat::byteswap(version);
      header_size = compat::byteswap(header_size);
    }

    memmove(tmp.data(), &version, sizeof(version));
    memmove(reinterpret_cast<uint8_t*>(tmp.data()) + sizeof(version), &header_size, sizeof(header_size));
    memmove(reinterpret_cast<uint8_t*>(tmp.data()) + sizeof(version) + sizeof(header_size), msgpk.data(), msgpk.size());
    memmove(reinterpret_cast<uint8_t*>(tmp.data()) + sizeof(version) + sizeof(header_size) + msgpk.size(), output->data(), output->size_in_bytes());

    *output = std::move(tmp);

    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    uint32_t version = *reinterpret_cast<uint32_t*>(input->data());
    if (compat::endian::native == compat::endian::big) {
      version = compat::byteswap(version);
    }
    if(version == 1) {
      uint64_t header_size = *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(input->data()) + sizeof(uint32_t));
      if (compat::endian::native == compat::endian::big) {
        header_size = compat::byteswap(header_size);
      }

      compat::string_view sv(
            reinterpret_cast<char*>(input->data()) + sizeof(uint32_t) + sizeof(uint64_t),
            header_size
          );
      nlohmann::json j = nlohmann::json::from_msgpack(sv);
      auto real_input = pressio_data::nonowning(
          pressio_byte_dtype,
          reinterpret_cast<uint8_t*>(input->data()) + sizeof(uint32_t) + sizeof(uint64_t) + header_size,
          {input->size_in_bytes() - (sizeof(uint32_t) + sizeof(uint64_t) + header_size)}
          );
      pressio_dtype dtype = j["t"].get<pressio_dtype>();
      auto dims = j["d"].get<std::vector<size_t>>();

      if(output->dtype() != dtype && dims != output->dimensions()) {
        *output = pressio_data::empty( dtype,dims);
      }

      set_name(j["n"].get<std::string>());
      set_options(j["o"].get<pressio_options>());

      auto saved_versions = j["c"].get<std::map<std::string, std::string>>();
      auto current_versions = versions();

      impl->decompress(&real_input, output);
    } else {
      return set_error(1, "unsuported version");
    }

    return 0;
  }

  std::map<std::string, std::string> versions() {
   std::map<std::string, std::string> m;
   m[get_name()] = version();
   return m;
  }

  void set_name_impl(std::string const& new_name) override {
    impl->set_name(new_name + "/" + impl->prefix());
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "manifest"; }

  pressio_options get_metrics_results_impl() const override {
    return impl->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<manifest_compressor_plugin>(*this);
  }

  pressio_compressor impl = compressor_plugins().build("noop");
  std::string impl_id = "noop";
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "manifest", []() {
  return compat::make_unique<manifest_compressor_plugin>();
});

} }

