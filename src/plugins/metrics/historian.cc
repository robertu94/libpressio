#include <mutex>
#include <sstream>
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include "std_compat/string_view.h"

namespace libpressio { namespace metrics { namespace historian {

class pressio_historian_metric: public libpressio_metrics_plugin {

  //these methods are "private" outside of this file, but need to be public
  //to be shared with the factory function below
  public:
  pressio_historian_metric()=default;
  pressio_historian_metric(pressio_historian_metric && lhs) noexcept:
      lock(), opts(std::exchange(lhs.opts, {})), idx(std::exchange(lhs.idx, {})),
      metrics_id(std::exchange(lhs.metrics_id, {})), metrics(std::exchange(lhs.metrics, {})),
      events(std::exchange(lhs.events, {})) {
      }
  pressio_historian_metric& operator=(pressio_historian_metric && lhs) noexcept {
    if (&lhs != this) return *this;
    opts = std::exchange(lhs.opts,{});
    idx = std::exchange(lhs.idx, {});
    metrics_id = std::exchange(lhs.metrics_id, {});
    metrics = std::exchange(lhs.metrics, {});
    events = std::exchange(lhs.events, {});
    return *this;
  }

  //historian is special in that it needs to offer a higher degree of thread safety to be useful
  //since it can record the number of times that a metrics plugin is cloned, which might happen
  //from a multi-threaded context (an exception to the normal thread safety rules of libpressio).
  //thus it needs an internal mutex to ensure consistently; but std::mutex is not copy-able meaning
  //that we need to default construct a new one, and then copy all of the other state.
  pressio_historian_metric(pressio_historian_metric const& lhs):
    lock(), opts(lhs.opts), idx(lhs.idx), metrics_id(lhs.metrics_id),
    metrics(lhs.metrics), events(lhs.events)
  {
    //I don't think we need to lock here because we guarantee that we lock elsewhere
    //before calling this method
  }
  pressio_historian_metric& operator=(pressio_historian_metric const& lhs) {
    if(&lhs != this) return *this;
    //I don't think we need to lock here because we guarantee that we lock elsewhere
    //before calling this method
    opts = lhs.opts;
    idx = lhs.idx;
    metrics_id = lhs.metrics_id;
    metrics = lhs.metrics;
    events = lhs.events;
    return *this;
  }

  private:
  const char* prefix() const override {
    return "historian";
  }
  int begin_get_options_impl() override {
    int rc = metrics->begin_get_options();
    return rc;
  }
  int end_get_options_impl(pressio_options const* opts) override {
    int ret = metrics->end_get_options(opts);
    if(events.on_get_options) {
      record();
    }
    return ret;
  }
  int begin_get_documentation_impl() override {
    int rc = metrics->begin_get_documentation();
    return rc;
  }
  int end_get_documentation_impl(pressio_options const& opts) override {
    int ret = metrics->end_get_documentation(opts);
    if(events.on_get_documentation) {
      record();
    }
    return ret;
  }
  int begin_get_configuration_impl() override {
    int rc = metrics->begin_get_configuration();
    return rc;
  }
  int end_get_configuration_impl(pressio_options const& opts) override {
    int ret = metrics->end_get_configuration(opts);
    if(events.on_get_configuration) {
      record();
    }
    return ret;
  }

  int begin_check_options_impl(pressio_options const* opts) override {
    int rc = metrics->begin_check_options(opts);
    return rc;
  }
  int end_check_options_impl(pressio_options const* opts, int rc) override {
    int ret = metrics->end_check_options(opts, rc);
    if(events.on_check_options) {
      record();
    }
    return ret;
  }

  int begin_set_options_impl(pressio_options const& opts) override {
    int rc = metrics->begin_set_options(opts);
    return rc;
  }
  int end_set_options_impl(pressio_options const& opts, int rc) override {
    int ret = metrics->end_set_options(opts, rc);
    if(events.on_set_options) {
      record();
    }
    return ret;
  }

  int begin_decompress_impl(pressio_data const* input, pressio_data const* output) override {
    int rc =  metrics->begin_decompress(input, output);
    return rc;
  }
  int end_decompress_impl(pressio_data const* input, pressio_data const* output, int rc) override {
    int ret = metrics->end_decompress(input, output, rc);
    if(events.on_decompress) {
      record();
    }
    return ret;
  }

  int begin_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    int rc =  metrics->begin_decompress_many(inputs, outputs);
    return rc;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    int ret = metrics->end_decompress_many(inputs, outputs, rc);
    if(events.on_decompress_many) {
      record();
    }
    return ret;
  }

  int begin_compress_impl(pressio_data const* input, pressio_data const* output) override {
    int rc =  metrics->begin_compress(input, output);
    return rc;
  }
  int end_compress_impl(pressio_data const* input, pressio_data const* output, int rc) override {
    int ret = metrics->end_compress(input, output, rc);
    if(events.on_compress) {
      record();
    }
    return ret;
  }

  int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    int rc =  metrics->begin_compress_many(inputs, outputs);
    return rc;
  }

  int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    int ret = metrics->end_compress_many(inputs, outputs, rc);
    if(events.on_compress_many) {
      record();
    }
    return ret;
  }

  void record() {
    std::lock_guard<std::mutex> guard(lock);
    std::stringstream ss;
    if(!name.empty()) {
      ss << get_name() << '/';
    }
    ss << idx;
    metrics->set_name(ss.str());
    opts.copy_from(metrics->get_metrics_results({}));
    idx++;
  }


  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    if (events.on_clone) {
      record();
    }
    return compat::make_unique<pressio_historian_metric>(*this);
  }
  int set_options(pressio_options const& opts) override {
    get_meta(opts, "historian:metrics", metrics_plugins(),  metrics_id, metrics);
    get(opts, "historian:idx", &idx);
    std::vector<std::string> events_str;
    if (get(opts, "historian:events", &events_str) == pressio_options_key_set) {
      events = event_hooks(events_str);
    }
    return 0;
  }
  pressio_options get_options() const override {
    pressio_options opts;
    set_meta(opts, "historian:metrics", metrics_id, metrics);
    set(opts, "historian:idx", idx);
    std::vector<std::string> events_str = static_cast<std::vector<std::string>>(events);
    set(opts, "historian:events", events_str);
    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set_meta_configuration(opts, "historian:metrics", metrics_plugins(), metrics);
    set(opts, "pressio:thread_safe", get_threadsafe(*metrics));
    set(opts, "pressio:stability", "unstable");
    const static std::vector<std::string> events_types {
       "check_options",
       "compress",
       "compress_many",
       "decompress",
       "decompress_many",
       "get_configuration",
       "get_documentation",
       "get_options",
       "set_options",
       "clone"
    };
    set(opts, "historian:events", events_types);
    set(opts, "predictors:requires_decompress", events.on_decompress || events.on_decompress_many);
    set(opts, "predictors:invalidate", std::vector<std::string>{});

    return opts;
  };
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set_meta_docs(opts, "historian:metrics", "what metrics should be recorded by the metric", metrics);
    set(opts, "pressio:description", "records metrics results after designated events");
    set(opts, "historian:idx", "the current index for this repetition");
    set(opts, "historian:events", "what events should trigger a record event");

    return opts;
  };
  pressio_options get_metrics_results(pressio_options const&)  override {
    return opts;
  }
  void set_name_impl(std::string const& new_name) override {
    metrics->set_name(new_name);
  }
  std::vector<std::string> children() const final {
      return { metrics->get_name() };
  }

  std::mutex lock;
  pressio_options opts;
  uint64_t idx = 0;
  std::string metrics_id = "noop";
  pressio_metrics metrics = metrics_plugins().build("noop");

  struct event_hooks {
    bool on_check_options = false;
    bool on_compress = false;
    bool on_compress_many = false;
    bool on_decompress = false;
    bool on_decompress_many = false;
    bool on_get_configuration = false;
    bool on_get_documentation = false;
    bool on_get_options = false;
    bool on_set_options = false;
    bool on_clone = false;

    event_hooks()=default;
    event_hooks(event_hooks const&)=default;
    event_hooks& operator=(event_hooks const&)=default;
    event_hooks(event_hooks &&)=default;
    event_hooks& operator=(event_hooks &&)=default;
    explicit event_hooks(std::vector<std::string> const& events):
      on_check_options(contains(events, "check_options")),
      on_compress(contains(events, "compress")),
      on_compress_many(contains(events, "compress_many")),
      on_decompress(contains(events, "decompress")),
      on_decompress_many(contains(events, "decompress_many")),
      on_get_configuration(contains(events, "get_configuration")),
      on_get_documentation(contains(events, "get_documentation")),
      on_get_options(contains(events, "get_options")),
      on_set_options(contains(events, "set_options")),
      on_clone(contains(events, "clone"))
    {}
    explicit operator std::vector<std::string>() const {
      std::vector<std::string> events;
      if(on_check_options) events.emplace_back("check_options");
      if(on_compress) events.emplace_back("compress");
      if(on_compress_many) events.emplace_back("compress_many");
      if(on_decompress) events.emplace_back("decompress");
      if(on_decompress_many) events.emplace_back("decompress_many");
      if(on_get_configuration) events.emplace_back("get_configuration");
      if(on_get_documentation) events.emplace_back("get_documentation");
      if(on_get_options) events.emplace_back("get_options");
      if(on_set_options) events.emplace_back("set_options");
      if(on_clone) events.emplace_back("clone");
      return events;
    }
    private:
    static bool contains(std::vector<std::string> const& c, compat::string_view item) {
      return std::find(c.begin(), c.end(), item) != c.end();
    }
  } events;
};

pressio_register registration (
    metrics_plugins(),
    "historian",
    []{
      return compat::make_unique<pressio_historian_metric>();
    }
);
} }}
