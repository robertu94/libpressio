#include "libpressio_ext/launch/external_launch.h"
#include "libpressio_ext/python/python.h"
#include <mutex>
#include "std_compat/memory.h"
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "pressio_compressor.h"


namespace libpressio { namespace python_launch {

namespace py = pybind11;
using namespace py::literals;

struct __attribute__((visibility ("hidden"))) libpressio_external_pybind_manager {
  libpressio_external_pybind_manager() {
  }
  ~libpressio_external_pybind_manager() {
  }

  py::scoped_interpreter guard;
};

static std::mutex libpressio_pybind_init_lock;
__attribute__((visibility ("default"))) std::shared_ptr<libpressio_external_pybind_manager> get_library() {
    std::lock_guard<std::mutex> guard(libpressio_pybind_init_lock);
    static std::weak_ptr<libpressio_external_pybind_manager> weak{};
    if(auto observed = weak.lock())
    {
        return observed;
    } else {
        auto library = std::make_shared<libpressio_external_pybind_manager>();
        weak = library;
        return library;
    }
}



struct __attribute__((visibility ("hidden"))) python_external_remote: public libpressio_launch_plugin {
  python_external_remote(std::shared_ptr<libpressio_external_pybind_manager>&& pybind_singleton):
    pybind_singleton(pybind_singleton) {}

extern_proc_results launch_impl(std::vector<std::string> const& full_command) const override {
      extern_proc_results results;

      try {
          py::str out = "", err = "";
          py::int_ ret = 0;
          auto locals = py::dict("cmd"_a=full_command, "stdout"_a=out, "stderr="_a=err, "ret"_a=ret);
          view_command({external_script});
          py::exec(external_script, py::globals(), locals);
          results.proc_stdout = locals["stdout"].cast<std::string>();
          results.proc_stderr = locals["stderr"].cast<std::string>();
          results.return_code = locals["ret"].cast<int>();
          results.error_code = 0;
      } catch(py::error_already_set const& ex) {
          results.proc_stdout = "external:api=5\n";
          results.proc_stderr = ex.what();
          results.return_code = 0;
          results.error_code = 1;
      }
      return results;
    }
  const char* prefix() const override {
    return "python";
  }

  int set_options_impl(pressio_options const& options) override {
    get(options, "python:script", &external_script);
    return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_serialized);
    set(options, "pressio:stability", "stable");
    return options;
  }

  pressio_options get_documentation_impl() const override {
    pressio_options options;
    set(options, "pressio:description", "request metrics from a python script");
    set(options, "python:script", "python script to execute");
    return options;
  }

  pressio_options get_options_impl() const override {
    pressio_options options;
    set(options, "python:script", external_script);
    return options;
  }

  std::unique_ptr<libpressio_launch_plugin> clone() const override {
    return compat::make_unique<python_external_remote>(*this);
  }

  std::string external_script;
  std::shared_ptr<libpressio_external_pybind_manager> pybind_singleton;
};

static pressio_register launch_spawn_plugin(launch_plugins(), "python", [](){
    return compat::make_unique<python_external_remote>(get_library());
});

}}
