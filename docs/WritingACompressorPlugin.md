# Writing a Compressor Plugin {#writingacompressor}

Libpressio supports adding custom compressor plugins.

To create a compressor plugin, simply create a subclass of `libpressio_compressor_plugin` that implements the compressor functions that you are interested in, and register the plugin with libpressio.

For example, let's create a compressor plugin that preforms a log transform as a preprocessing step to compression, and does a exponential transformation as a post processing step to decompression.  Some scientific data compresses much better in the log domain than in the normal domain.

First you will need to include several headers.

```cpp
#include <cmath> //for exp and log
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
```

First we are going to delegate most of the methods to the underlying compressor.
To do so, we are going to do this in one of the most naive was possible and accept the compressor as a constructor argument.
Then we are going to provide overrides that pass each of the methods to the underlying compressor.

Notice the function `check_error`.
Since we are using composition rather than inheritance, we need to propagate error messages to our wrapper.
The `check_error` function simply check for the error condition in the underlying function and propagates the error if these is one.

Also notice that we are using the `_impl` versions of the various methods.
This is because the base class is doing some work to provide metrics functionality and basic error checking that we want to use.

```cpp
class log_transform : public libpressio_compressor_plugin {
  public:
  log_transform(): compressor(nullptr) {}
  log_transform(pressio_compressor&& comp): compressor(std::move(comp)) {}


  //TODO compress and decompress, see below


  //getting and setting options/configuration
  pressio_options get_options_impl() const override {
    auto options =  compressor.plugin->get_options();
    options.set("log:compressor", (void*)&compressor);
    return options;
  }
  int set_options_impl(pressio_options const& options) override {
    //if the user hasn't set the compressor yet, fail
    if(!compressor) return invalid_compressor();
    int rc = check_error(compressor.plugin->set_options(options));

    //try to extract the compressor plugin from the option
    void* tmp;
    if(options.get("log:compressor", &tmp) == pressio_options_key_set) {
      compressor = std::move(*(pressio_compressor*)tmp);
    }

    return rc;
  }
  pressio_options get_configuration_impl() const override {
    //if the user hasn't set the compressor yet, fail
    if(!compressor) return pressio_options();
    return compressor.plugin->get_configuration();
  }
  int check_options_impl(pressio_options const& options) override {
    //if the user hasn't set the compressor yet, fail
    if(!compressor) return invalid_compressor();
    return check_error(compressor.plugin->check_options(options));
  }

  //getting version information
  const char* prefix() const override {
    return "log";
  }
  const char* version() const override {
    if(!compressor) return "";
    return compressor.plugin->version();
  }
  int major_version() const override {
    if(!compressor) return -1;
    return compressor.plugin->major_version();
  }
  int minor_version() const override {
    if(!compressor) return -1;
    return compressor.plugin->minor_version();
  }
  int patch_version() const override {
    if(!compressor) return -1;
    return compressor.plugin->patch_version();
  }

  private:
  int check_error(int rc) { 
    if(rc) {
      set_error(
          compressor.plugin->error_code(),
          compressor.plugin->error_msg()
          );
    }
    return rc;
  }
  int invalid_compressor() { return set_error(-1, "compressor must be set"); };
  pressio_compressor compressor;
};
```
Now to the harder part, writing compress and decompress.


First we can write our log and exponential transform functions.

```cpp
namespace {
  struct log_fn{
    template <class T>
    pressio_data operator()(T* begin, T* end) {
        pressio_data log_data = pressio_data::clone(*input);
        auto output_it = reinterpret_cast<T*>(log_data.data());
        std::transform(begin, end, output_it, [](T i){ return std::log(i); });
        return log_data;
    }
    pressio_data const* input;
  };

  struct exp_fn{
    template <class T>
    pressio_data operator()(T* begin, T* end) {
        pressio_data log_data = pressio_data::clone(*output);
        auto output_it = reinterpret_cast<T*>(log_data.data());
        std::transform(begin, end, output_it, [](T i){ return std::exp(i); });
        return log_data;
    }
    pressio_data const* output;
  };
}
```

We will use these methods using the `pressio_data_for_each` API to preform the transform
Now, we can write compress and decompress:


```cpp
  //compress and decompress
  int compress_impl(pressio_data const* input, pressio_data* output) override {
    if(!compressor) return invalid_compressor();
    pressio_data log_input = pressio_data_for_each<pressio_data>(*input, log_fn{input});
    return check_error(compressor.plugin->compress(&log_input, output));
  }

  int decompress_impl(pressio_data const* input, pressio_data* output) override {
    if(!compressor) return invalid_compressor();
    int rc =  compressor.plugin->decompress(input, output);
    *output = pressio_data_for_each<pressio_data>(*output, exp_fn{output});
    return check_error(rc);
  }
```

We finally register the library with libpressio:


```cpp
static pressio_register X(compressor_plugins(), "log", [](){ return std::make_unique<log_transform>();});
```

High quality compressor modules may be accepted into libpressio.  Contributed modules should be placed in to
the `src/plugins/compressors/` directory and added to the main CMakeLists.txt.  If the compressor plugin 
requires an external dependency, it should be hidden behind a configuration option.

