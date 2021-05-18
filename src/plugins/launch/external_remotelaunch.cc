#include "libpressio_ext/launch/external_launch.h"
#include <mutex>
#include "std_compat/memory.h"
#include <nlohmann/json.hpp>
#include "pressio_compressor.h"
#include <curl/curl.h>
#include <curl/easy.h>

static size_t write_to_std_string(char* txt, size_t size, size_t nelms, void* user_data) {
  std::string* user_str = reinterpret_cast<std::string*>(user_data);
  user_str->append(txt, size*nelms);
  return size*nelms;
}

static std::mutex libpressio_curl_init_lock;
struct libpressio_external_curl_manager {
  libpressio_external_curl_manager() {
    curl_global_init(CURL_GLOBAL_ALL);
  }
  ~libpressio_external_curl_manager() {
    curl_global_cleanup();
  }

  static std::shared_ptr<libpressio_external_curl_manager> get_library() {
    std::lock_guard<std::mutex> guard(libpressio_curl_init_lock);
    static std::weak_ptr<libpressio_external_curl_manager> weak{};
    if(auto observed = weak.lock())
    {
      return observed;
    } else {
      auto library = std::make_shared<libpressio_external_curl_manager>();
      weak = library;
      return library;
    }
  }
};



struct external_remote: public libpressio_launch_plugin {
  external_remote(std::shared_ptr<libpressio_external_curl_manager>&& curl_singleton):
    curl_singleton(curl_singleton) {}

extern_proc_results launch(std::vector<std::string> const& full_command) const override {
      extern_proc_results results;
      nlohmann::json request;
      request["args"] = full_command;
      std::string request_str = request.dump();
      std::string response_str;
      char errbuf[CURL_ERROR_SIZE] = {0};

      CURLcode ret;
      CURL *hnd;
      curl_slist* headers = nullptr;
      headers = curl_slist_append(headers, "Content-Type: application/json");
      hnd = curl_easy_init();
      curl_easy_setopt(hnd, CURLOPT_BUFFERSIZE, 102400L);
      curl_easy_setopt(hnd, CURLOPT_URL, connection_string.c_str());
      curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
      curl_easy_setopt(hnd, CURLOPT_USERAGENT, "curl/7.72.0");
      curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
      curl_easy_setopt(hnd, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_2TLS);
      curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);
      curl_easy_setopt(hnd, CURLOPT_POST, 1L);
      curl_easy_setopt(hnd, CURLOPT_POSTFIELDS, request_str.c_str());
      curl_easy_setopt(hnd, CURLOPT_POSTFIELDSIZE, request_str.size());
      curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, &write_to_std_string);
      curl_easy_setopt(hnd, CURLOPT_WRITEDATA, &response_str);
      curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(hnd, CURLOPT_ERRORBUFFER, errbuf);

      ret = curl_easy_perform(hnd);

      if(ret != CURLE_OK) {
        results.error_code = ret;
        if(strlen(errbuf)) {
          results.proc_stderr = std::string(errbuf);
        } else {
          results.proc_stderr = curl_easy_strerror(ret);
        }
      } else {
        try {
          nlohmann::json response = nlohmann::json::parse(response_str);
          results.proc_stdout = response["stdout"].get<std::string>();
          results.proc_stderr = response["stderr"].get<std::string>();
          results.return_code = response["return_code"].get<int>();
        } catch(nlohmann::json::exception const& e) {
          results.proc_stdout = "";
          results.proc_stderr = response_str + "\n\n" +  e.what();
          results.error_code = -1;
        }
      }

      curl_easy_cleanup(hnd);
      curl_slist_free_all(headers);
      hnd = NULL;


      return results;
    }
  const char* prefix() const override {
    return "remote";
  }

  int set_options(pressio_options const& options) override {
    get(options, "external:connection_string", &connection_string);
    return 0;
  }

  struct pressio_options get_configuration() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "stable");
    return options;
  }

  pressio_options get_documentation_impl() const override {
    pressio_options options;
    set(options, "pressio:description", "request metrics from a remote server");
    set(options, "external:connection_string", "curl connection string");
    return options;
  }

  pressio_options get_options() const override {
    pressio_options options;
    set(options, "external:connection_string", connection_string);
    return options;
  }

  std::unique_ptr<libpressio_launch_plugin> clone() const override {
    return compat::make_unique<external_remote>(*this);
  }

  std::string connection_string;
  std::shared_ptr<libpressio_external_curl_manager> curl_singleton;
};

static pressio_register launch_spawn_plugin(launch_plugins(), "remote", [](){
    return compat::make_unique<external_remote>(
        libpressio_external_curl_manager::get_library()
        );
});
