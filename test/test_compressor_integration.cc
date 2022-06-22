#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <numeric>
#include <libpressio_ext/cpp/libpressio.h>
#include <libpressio_ext/launch/external_launch.h>

std::set<std::tuple<std::string, std::string>> skip_list {
  {"SZauto", "3d float zeros"},
  {"qoz", "1d float"},
  {"qoz", "1d int"},
  {"qoz", "3d int"},
  {"qoz", "3d float zeros"},
};

template <class Func>
std::shared_ptr<const pressio_data> from_function(std::vector<size_t> const& dims, Func&& func) {
  std::vector<size_t> current(dims.size(), 0);

  using T = decltype(func(current));
  pressio_data data = pressio_data::owning(pressio_dtype_from_type<T>(), dims);
  T* ptr = reinterpret_cast<T*>(data.data());


  bool done = data.size_in_bytes() == 0;
  while(!done) {

    *ptr++ = func(current);

    //update
    size_t idx = 0;
    bool keep_going = false;
    do {
      keep_going = false;
      current[idx]++;
      if(current[idx] == dims[idx]) {
        current[idx] = 0;
        idx++;
        keep_going = true;
      }
      if(idx == dims.size()) {
        keep_going = false;
        done = true;
      }
    } while(keep_going);
  }

  return std::make_shared<const pressio_data>(std::move(data));
}

std::map<std::string, std::shared_ptr<const pressio_data>>& data_test_cases() {
  static std::map<std::string, std::shared_ptr<const pressio_data>> cases{
    {"1d float", from_function({500}, [](std::vector<size_t> const& i) -> float {
        return float( i[0] + 0.5f ); })},
    {"1d int", from_function({500}, [](std::vector<size_t> const& i) -> int {
        return int( i[0] ); })},
    {"2d float", from_function({500,500}, [](std::vector<size_t> const& i) -> float {
        return float(i[0] + 500*i[1] ) + 0.5f; })},
    {"3d float", from_function({62,62,62}, [](std::vector<size_t> const& i) -> float {
        return float((i[0] - 31 ) + (i[1] - 31) + (i[2] - 31)) + 0.5f; })},
    {"3d float zeros", from_function({62,62,62}, [](std::vector<size_t> const& i) -> float {
        return float(0);})},
    {"2d int", from_function({500,500}, [](std::vector<size_t> const& i) -> int {
        return int(i[0] + 500*i[1]); })},
    {"3d int", from_function({62,62,62}, [](std::vector<size_t> const& i) -> int {
        return int((i[0] - 31 ) + (i[1] - 31) + (i[2] - 31));})},
    {"2d 0-1 float", from_function({500,500}, [](std::vector<size_t> const& i) -> float {
        return float(i[0] + 500*i[1]) / 250000.0f; })},
  };
  return cases;
}

std::vector<std::string> data_test_cases_names() {
  std::vector<std::string> v;
  auto& cases = data_test_cases();
  std::transform(
      std::begin(cases), std::end(cases),
      std::back_inserter(v),
      [](decltype(*cases.begin()) entries) {
        return entries.first;
      }
      );
  return v;
}


class PressioCompressorIntegrationConfigOnly : public testing::TestWithParam<std::string> {
  void SetUp() {
    compressor = pressio().get_compressor(GetParam());
  }

  protected:
  pressio_compressor compressor;
};

class PressioMetricsIntegrationConfigOnly : public testing::TestWithParam<std::string> {
  void SetUp() {
    metrics = metrics_plugins().build(GetParam());
  }

  protected:
  pressio_metrics metrics;
};
class PressioIOIntegrationConfigOnly : public testing::TestWithParam<std::string> {
  void SetUp() {
    io = io_plugins().build(GetParam());
  }

  protected:
  pressio_io io;
};
class PressioLaunchIntegrationConfigOnly : public testing::TestWithParam<std::string> {
  void SetUp() {
    launcher = launch_plugins().build(GetParam());
  }

  protected:
  pressio_launcher launcher;
};



class PressioCompressorIntegrationConfigAndData : public testing::TestWithParam<std::tuple<std::string, std::string>> {
  void SetUp() {
    compressor = pressio().get_compressor(std::get<0>(GetParam()));
    data = data_test_cases()[std::get<1>(GetParam())];
  }

  protected:
  pressio_compressor compressor;
  std::shared_ptr<const pressio_data> data;
};


template <class Registry>
std::vector<std::string> supported(Registry const& registry) {
  std::vector<std::string> ids;
  std::transform(std::begin(registry), std::end(registry), std::back_inserter(ids), 
      [](decltype(*registry.begin()) i) {
        return i.first;
      });
  return ids;
}



TEST_P(PressioCompressorIntegrationConfigOnly, CompressorIsConstructable) {
  ASSERT_NE(compressor.plugin, nullptr);
}

template <class GetNames>
void test_configuarable_is_documented(pressio_configurable const& configurable, GetNames&& get_names){
  auto docs = configurable.get_documentation();
  auto options = get_names();
  std::set<std::string> names;
  std::set<std::string> documented_names;

  std::transform(
      std::begin(docs), std::end(docs),
      std::inserter(documented_names, documented_names.end()),
      [](decltype(*docs.begin()) i) {
        return i.first;
  });

  std::transform(
      std::begin(options), std::end(options),
      std::inserter(names, names.end()),
      [](decltype(*options.begin()) i) {
        return i.first;
  });


  std::set<std::string> undocumented_names;
  std::set_difference(
      names.begin(), names.end(),
      documented_names.begin(), documented_names.end(),
      std::inserter(undocumented_names, undocumented_names.end()));

  EXPECT_EQ(docs.key_status("pressio:description"), pressio_options_key_set);
  EXPECT_THAT(undocumented_names, testing::IsEmpty()) << "\nknown:\n" << options << "\ndocumented:\n" << docs;
}

TEST_P(PressioCompressorIntegrationConfigOnly, IsDocumented) {
  test_configuarable_is_documented(*compressor, [this]{
        pressio_options opts;
        opts.copy_from(compressor->get_configuration());
        opts.copy_from(compressor->get_options());
        return opts;
      });
}
TEST_P(PressioMetricsIntegrationConfigOnly, IsDocumented) {
  test_configuarable_is_documented(*metrics, [this]{
        pressio_options opts;
        opts.copy_from(metrics->get_metrics_results({}));
        opts.copy_from(metrics->get_configuration());
        opts.copy_from(metrics->get_options());
        return opts;
      });
}
TEST_P(PressioIOIntegrationConfigOnly, IsDocumented) {
  test_configuarable_is_documented(*io, [this]{
        pressio_options opts;
        opts.copy_from(io->get_configuration());
        opts.copy_from(io->get_options());
        return opts;
      });
}
TEST_P(PressioLaunchIntegrationConfigOnly, IsDocumented) {
  test_configuarable_is_documented(*launcher, [this]{
        pressio_options opts;
        opts.copy_from(launcher->get_configuration());
        opts.copy_from(launcher->get_options());
        return opts;
      });
}

void test_has_configuration(pressio_configurable& c) {
  auto config = c.get_configuration();

  int32_t thread_safe = 9;
  std::string stability;
  EXPECT_EQ(config.key_status("pressio:thread_safe"), pressio_options_key_set) << config;
  EXPECT_EQ(config.key_status("pressio:stability"), pressio_options_key_set) << config;
  EXPECT_EQ(config.get("pressio:thread_safe", &thread_safe), pressio_options_key_set);
  EXPECT_EQ(config.get("pressio:stability", &stability), pressio_options_key_set) << config;
}
TEST_P(PressioCompressorIntegrationConfigOnly, HasConfiguration) {
  test_has_configuration(*compressor);
}
TEST_P(PressioMetricsIntegrationConfigOnly, HasConfiguration) {
  test_has_configuration(*metrics);
}
TEST_P(PressioIOIntegrationConfigOnly, HasConfiguration) {
  test_has_configuration(*io);
}
TEST_P(PressioLaunchIntegrationConfigOnly, HasConfiguration) {
  test_has_configuration(*launcher);
}

TEST_P(PressioCompressorIntegrationConfigAndData, Compress) {
  if(skip_list.find(GetParam()) != skip_list.end()) GTEST_SKIP();
  pressio_data output = pressio_data::empty(pressio_byte_dtype, {});
  pressio_data decompress = pressio_data::clone(*this->data);
  int rc = compressor->compress(this->data.get(), &output);
  if(!rc) {
    compressor->decompress(&output, &decompress);
  }
}

INSTANTIATE_TEST_SUITE_P(AllCompressors,
    PressioCompressorIntegrationConfigOnly,
    testing::ValuesIn(supported(compressor_plugins()))
    );
INSTANTIATE_TEST_SUITE_P(AllMetrics,
    PressioMetricsIntegrationConfigOnly,
    testing::ValuesIn(supported(metrics_plugins()))
    );
INSTANTIATE_TEST_SUITE_P(AllIO,
    PressioIOIntegrationConfigOnly,
    testing::ValuesIn(supported(io_plugins()))
    );
INSTANTIATE_TEST_SUITE_P(AllIO,
    PressioLaunchIntegrationConfigOnly,
    testing::ValuesIn(supported(launch_plugins()))
    );

INSTANTIATE_TEST_SUITE_P(AllCompressorsWithData,
    PressioCompressorIntegrationConfigAndData,
    testing::Combine(
      testing::ValuesIn(supported(compressor_plugins())),
      testing::ValuesIn(data_test_cases_names())
    ));


template <class Registry>
void test_no_matching_descriptions(Registry const& registry) {
  std::map<std::string, std::vector<std::string>> ids_by_desc;
  for (auto& entry : registry) {
    std::string description;
    auto compressor = entry.second();
    compressor->get_documentation().get("pressio:description", &description);
    ids_by_desc[description].emplace_back(entry.first);
  }

  for (auto const& i : ids_by_desc) {
    EXPECT_THAT(i.second,  testing::SizeIs(1)) << i.first;
  }
}
TEST(AllCompressors, NoMatchingDescriptions) {
  test_no_matching_descriptions(compressor_plugins());
}
TEST(AllIO, NoMatchingDescriptions) {
  test_no_matching_descriptions(io_plugins());
}
TEST(AllMetrics, NoMatchingDescriptions) {
  test_no_matching_descriptions(metrics_plugins());
}
TEST(AllLaunch, NoMatchingDescriptions) {
  test_no_matching_descriptions(launch_plugins());
}

template <class Registry>
void test_no_matching_prefixes(Registry const& registry) {
  std::map<std::string, std::vector<std::string>> ids_by_desc;
  for (auto& entry : compressor_plugins()) {
    std::string description;
    auto compressor = entry.second();
    auto prefix = compressor->prefix();
    ids_by_desc[prefix].emplace_back(entry.first);
  }

  for (auto const& i : ids_by_desc) {
    EXPECT_THAT(i.second,  testing::SizeIs(1)) << i.first;
  }
}

TEST(AllCompressors, NoMatchingPrefixes) {
  test_no_matching_prefixes(compressor_plugins());
}
TEST(AllMetrics, NoMatchingPrefixes) {
  test_no_matching_prefixes(metrics_plugins());
}
TEST(AllIO, NoMatchingPrefixes) {
  test_no_matching_prefixes(io_plugins());
}
TEST(AllLaunch, NoMatchingPrefixes) {
  test_no_matching_prefixes(launch_plugins());
}


MATCHER_P(ElementOf, container, std::string(negation ? "isn't": "is") + " an element of" + testing::PrintToString(container)) {
  return std::find(std::begin(container), std::end(container), arg) != std::end(container);
}

TEST(AllCompressors, NoSkipsOnNonExperimentalCompressors) {
  std::vector<std::string> nonexperimental_compressors;
  for (auto const& i : compressor_plugins()) {
    std::string stability;
    i.second()->get_configuration().get("pressio:stability", &stability);
    if(stability != "experimental") {
      nonexperimental_compressors.emplace_back(i.first);
    }
  }

  for (auto const& i : skip_list) {
    EXPECT_THAT(std::get<0>(i), testing::Not(ElementOf(nonexperimental_compressors)));
  }
}
