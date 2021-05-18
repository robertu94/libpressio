# Writing a User Metrics Plugin {#writingametric}

libpressio supports adding custom metrics plugins.


To create a metrics plugin, simply create a subclass of `libpressio_metrics_plugin` that implements the check that you are interested in, and register the plugin with libpressio.

For example, let's create a plugin that counts the number of compressions of each data type that the user preforms.

First we will include a number of required headers:

```cpp
//required for all plugins
#include "libpressio_ext/cpp/metrics.h" // provides the libpressio_metrics_plugin
#include "libpressio_ext/cpp/options.h" // provides pressio_options and related methods
#include "libpressio_ext/cpp/pressio.h" // for the plugin registry

//required for this plugin
#include "libpressio_ext/cpp/data.h" // to be able to access ->dtype()
#include <memory> //for make_unique (in c++14 or later)
#include <map> //for the map that counts what data type was used

```

Next, we need to write our class.  Since we want to count the number of data buffers of each type that we compress, we will hook the `begin_compress_impl` method.  Alternatively we could also hook the `end_compress_impl` method.  Once we have the counts, we report them out using the `get_metrics_results` function.  We do this like so:

```cpp
class counting_metric: public libpressio_metrics_plugin {
  counting_metric() {
    //operator[] is non-const, so explicits instantiate each of the values we need
    counts[pressio_int8_dtype] = 0;
    counts[pressio_int16_dtype] = 0;
    counts[pressio_int32_dtype] = 0;
    counts[pressio_int64_dtype] = 0;
    counts[pressio_uint8_dtype] = 0;
    counts[pressio_uint16_dtype] = 0;
    counts[pressio_uint32_dtype] = 0;
    counts[pressio_uint64_dtype] = 0;
    counts[pressio_float_dtype] = 0;
    counts[pressio_double_dtype] = 0;
    counts[pressio_byte_dtype] = 0;
  }

  //increment the count at the beginning of each compress
  void begin_compress_impl(pressio_data const* input, pressio_data const*) override {
    counts[input->dtype()]++;
  }

  //return an options structure containing the metrics we care about
  //notice how the namespaced matches the name of the plugin that is registered below
  pressio_options get_metrics_results() const override {
    pressio_options opts;
    set(opts, "counts:int8", counts.at(pressio_int8_dtype));
    set(opts, "counts:int16", counts.at(pressio_int16_dtype));
    set(opts, "counts:int32", counts.at(pressio_int32_dtype));
    set(opts, "counts:int64", counts.at(pressio_int64_dtype));
    set(opts, "counts:uint8", counts.at(pressio_uint8_dtype));
    set(opts, "counts:uint16", counts.at(pressio_uint16_dtype));
    set(opts, "counts:uint32", counts.at(pressio_uint32_dtype));
    set(opts, "counts:uint64", counts.at(pressio_uint64_dtype));
    set(opts, "counts:float", counts.at(pressio_float_dtype));
    set(opts, "counts:double", counts.at(pressio_double_dtype));
    set(opts, "counts:byte", counts.at(pressio_byte_dtype));
    return opts;
  }

  const char* prefix() const {
    return "counts";
  }

  std::map<pressio_dtype, unsigned int> counts;
};
```

Finally, we will register the plugin in the under the names "counts" in the metrics plugging registry

```cpp
static pressio_register X(metrics_plugins(), "counts", [](){ return std::make_unique<counting_metric>(); });
```

Then a user of the library can then ask libpressio to construct their new plugin as normal.
But what if our metrics modules takes arguments?
Instead of registering it directly, the user can instantiate it manually and combine it with others from the library using the `make_m_composite` method.

If your metric is suitably general and well implemented, you can contribute it back to libpressio.
Metrics should be added to the `src/plugins/metrics/` directory and to the main `CMakeLists.txt`.  If the metric requires additional dependencies beyond the standard library, it should be hidden behind a configuration flag that enables its use and disabled by default.

This full example can be found in the `test/test_regsiter_metrics.cc` file, and the full documentation of the allowed hooks and their arguments can be found in the documentation for `libpressio_metrics_plugin`.
