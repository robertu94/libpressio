#include <libpressio_ext/cpp/printers.h>
#include <gtest/gtest.h>

#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

namespace libpressio { namespace compressors { namespace highlevel_mock_ns {

class highlevel_mock_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "highlevel_mock:i32", i32);
    set(options, "highlevel_mock:f32", f32);
    set(options, "highlevel_mock:s", s);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"()");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "highlevel_mock:i32", &i32);
    get(options, "highlevel_mock:f32", &f32);
    get(options, "highlevel_mock:s", &s);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
      return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
      return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "highlevel_mock"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<highlevel_mock_compressor_plugin>(*this);
  }

  int32_t i32 = 0;
  float f32 = 0;
  std::string s = "";
};

pressio_register compressor_many_fields_plugin(compressor_plugins(), "highlevel_mock", []() {
  return compat::make_unique<highlevel_mock_compressor_plugin>();
});

} } }

void test_contains(pressio_options const& actual, pressio_options const& expected) {
    for (auto const& i : expected) {
        ASSERT_EQ(actual.key_status(i.first), pressio_options_key_set);
        ASSERT_EQ(actual.get(i.first).type(), i.second.type());
        ASSERT_EQ(actual.get(i.first), i.second);
    }
}

TEST(HighLevel, OptionsCast) {
    pressio library;

    pressio_options expected {
        {"highlevel_mock:s", "testing"},
        {"highlevel_mock:i32", int32_t{3}},
        {"highlevel_mock:f32", float{2.0f}},
    };
    
    {
        //all is early
        pressio_compressor c = library.get_compressor("highlevel_mock");
        c->cast_options(
                {
                    {"highlevel_mock:s", "testing"},
                    {"highlevel_mock:i32", int32_t{3}},
                    {"highlevel_mock:f32", float{2.0f}},
                },
                {
                }
        );
        test_contains(c->get_options(), expected);
    }

    {
        //all is late
        pressio_compressor c = library.get_compressor("highlevel_mock");
        c->cast_options(
                {
                },
                {
                    {"highlevel_mock:s", "testing"},
                    {"highlevel_mock:i32", int32_t{3}},
                    {"highlevel_mock:f32", float{2.0f}},
                }
        );
        test_contains(c->get_options(), expected);
    }

    {
        //all are early strings
        pressio_compressor c = library.get_compressor("highlevel_mock");
        c->cast_options(
                {
                    {"highlevel_mock:s", "testing"},
                    {"highlevel_mock:i32", "3"},
                    {"highlevel_mock:f32", "2.0"},
                },
                {
                }
        );
        test_contains(c->get_options(), (pressio_options{
            {"highlevel_mock:s", "testing"},
            {"highlevel_mock:i32", int32_t{0}},
            {"highlevel_mock:f32", float{0}},
        }));
    }

    {
        //all are late strings
        pressio_compressor c = library.get_compressor("highlevel_mock");
        c->cast_options(
                {
                },
                {
                    {"highlevel_mock:s", "testing"},
                    {"highlevel_mock:i32", "3"},
                    {"highlevel_mock:f32", "2.0"},
                }
        );
        test_contains(c->get_options(), expected);
    }
}

TEST(HighLevel, OptionsInvalid) {
    pressio library;
    pressio_compressor c = library.get_compressor("highlevel_mock");
    void* ptr = nullptr;
    int rc = c->cast_options(
            {
            },
            {
                {"highlevel_mock:i32", ptr}, //invalid type
            }
    );
    ASSERT_NE(rc, 0);
    ASSERT_NE(c->error_code(), 0);
    ASSERT_NE(c->error_msg(), "");
}
TEST(HighLevel, Missing) {
    pressio library;
    pressio_compressor c = library.get_compressor("highlevel_mock");
    int rc = c->cast_options(
            {
            },
            {
                {"highlevel_mock:s", "testing"},
                {"highlevel_mock:foobar", "3"}, //non-existent option
                {"highlevel_mock:f32", "2.0"},
            }
    );
    ASSERT_NE(rc, 0);
    ASSERT_NE(c->error_code(), 0);
    ASSERT_NE(c->error_msg(), "");
}
