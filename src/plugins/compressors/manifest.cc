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
#include <iostream>


namespace libpressio { namespace manifest_ns {

class manifest_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "manifest:compressor", impl_id, impl);
    set(options, "manifest:lineage", lineage);
    set(options, "manifest:record_lineage_on_decompress", record_on_decompress);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "manifest:compressor", compressor_plugins(), impl);
    set(options, "pressio:thread_safe", get_threadsafe(*impl));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "manifest:compressor", R"(compressor to use to decompress)", impl);
    set(options, "pressio:description", R"(compressor plugin that records meta data from compression to enable reconstruction)");
    set(options, "manifest:lineage", R"(header for the last compression operation)");
    set(options, "manifest:record_lineage_on_decompress", R"(update manifest:lineage when decompress is called)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "manifest:compressor", compressor_plugins(), impl_id, impl);
    get(options, "manifest:lineage", &lineage);
    get(options, "manifest:record_lineage_on_decompress", &record_on_decompress);
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

    std::cout << j << std::endl;
    std::vector<uint8_t> msgpk = nlohmann::json::to_msgpack(j);

    int rc = impl->compress(input, output);
    if(rc) {
      return set_error(impl->error_code(), impl->error_msg());
    }
    this->header_size = msgpk.size() + sizeof(uint32_t) + sizeof(uint64_t);
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
      this->header_size = header_size + sizeof(uint32_t) + sizeof(uint64_t);

      compat::string_view sv (
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
      
      int val_rc = validate_versions(saved_versions, current_versions);
      if(val_rc < 0) {
        //warning
        std::stringstream ss;
        ss << "some versions didn't match old={";
        for (auto const& i : saved_versions) {
          ss << '{' << i.first << ',' << i.second << '}';
        }
        ss << "} new={";
        for (auto const& i : current_versions) {
          ss << '{' << i.first << ',' << i.second << '}';
        }
        ss << "}";
        set_error(1, ss.str());
      } 
      if(val_rc > 0) {
        return error_code();
      }

      impl->decompress(&real_input, output);
      if(record_on_decompress) {
        lineage = static_cast<std::string>(sv);
      }
    } else {
      return set_error(1, "unsuported version");
    }

    return 0;
  }

  std::map<std::string, std::string> versions() {
   auto current_name = get_name();
   set_name("p");
   std::map<std::string, std::string> m;

   auto config = get_configuration();
   for (auto const& e : config) {
     std::string const& key = e.first;
     pressio_option const& value = e.second;

     auto const& path = key.substr(0, key.find(':'));
     auto const& var = key.substr(key.find(':')+1);

     if (var == "pressio:version_major") {
       m[path] = m[path] + value.as(pressio_option_charptr_type, pressio_conversion_special).get_value<std::string>();
     } else if (var =="pressio:version_epoch") {
       m[path] = value.as(pressio_option_charptr_type, pressio_conversion_special).get_value<std::string>() + ":" + m[path];
     }
   }

   set_name(current_name);
   return m;
  }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
      impl->set_name(new_name + "/" + impl->prefix());
    } else {
      impl->set_name(new_name);
    }
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "manifest"; }

  pressio_options get_metrics_results_impl() const override {
    auto options =  impl->get_metrics_results();
    set(options, "manifest:header_size", header_size);

    return options;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<manifest_compressor_plugin>(*this);
  }

private:
  int validate_versions(std::map<std::string, std::string> const& old_versions,
    std::map<std::string, std::string> const& loaded_versions) {
    int ret = 0;
    for (auto const& old : old_versions) {
      auto new_it = loaded_versions.find(old.first);
      if(new_it != loaded_versions.end()) {
        if(old.second != new_it->second) {
          //version mismatch, issue a warning
          ret--;
        }
      } else {
        //missing version, something is incompatible
        return 1;
      }
    }

    return ret;
  }

  pressio_compressor impl = compressor_plugins().build("noop");
  std::string impl_id = "noop";
  std::string lineage = "";
  uint64_t header_size = 0;
  bool record_on_decompress = false;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "manifest", []() {
  return compat::make_unique<manifest_compressor_plugin>();
});

} }

