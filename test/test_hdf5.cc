#include <vector>
#include <numeric>
#include "libpressio_ext/io/hdf5.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/iterator.h"
#include "pressio_data.h"
#include "gtest/gtest.h"

class PressioIOHDFTests: public ::testing::Test {

  void SetUp() override {
    tmp_fd = mkstemps(test_file, 3);
  }
  void TearDown()  override{
    close(tmp_fd);
  }
  
  protected:
  char test_file[14] = "testXXXXXX.h5";
  int tmp_fd;
};

TEST_F(PressioIOHDFTests, read_and_write) {
  std::vector<float> floats(3*3*3);
  std::vector<size_t> dims{3,3,3};
  std::iota(std::begin(floats), std::end(floats), 0.0);

  pressio_data* data = pressio_data_new_nonowning(pressio_float_dtype, floats.data(), dims.size(), dims.data());


  {
    auto ret = pressio_io_data_path_h5write(data, test_file, "testing");
    EXPECT_EQ(ret, 0);
  }

  {
    pressio_data* result = pressio_io_data_path_h5read(test_file, "testing");
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(pressio_float_dtype, pressio_data_dtype(result));
    ASSERT_EQ(3, pressio_data_num_dimensions(result));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 0));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 1));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 2));
    std::vector<float> result_data(3*3*3);
    size_t size_in_bytes;
    void* result_ptr = pressio_data_ptr(result, &size_in_bytes);
    memcpy(result_data.data(), result_ptr, size_in_bytes);
    EXPECT_EQ(floats, result_data);
    pressio_data_free(result);
  }

  pressio_data_free(data);
}

TEST_F(PressioIOHDFTests, write_twice) {
  std::vector<float> floats(3*3*3);
  std::vector<size_t> dims{3,3,3};
  std::iota(std::begin(floats), std::end(floats), 0.0);

  pressio_data* data = pressio_data_new_nonowning(pressio_float_dtype, floats.data(), dims.size(), dims.data());


  {
    auto ret = pressio_io_data_path_h5write(data, test_file, "testing");
    EXPECT_EQ(ret, 0);
  }

  {
    pressio_data* result = pressio_io_data_path_h5read(test_file, "testing");
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(pressio_float_dtype, pressio_data_dtype(result));
    ASSERT_EQ(3, pressio_data_num_dimensions(result));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 0));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 1));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 2));
    std::vector<float> result_data(3*3*3);
    size_t size_in_bytes;
    void* result_ptr = pressio_data_ptr(result, &size_in_bytes);
    memcpy(result_data.data(), result_ptr, size_in_bytes);
    ASSERT_EQ(result_data.front(), 0.0f);
    EXPECT_EQ(floats, result_data);
    pressio_data_free(result);
  }

  std::iota(compat::rbegin(floats), compat::rend(floats), 0);


  {
    auto ret = pressio_io_data_path_h5write(data, test_file, "testing");
    EXPECT_EQ(ret, 0);
  }

  {
    pressio_data* result = pressio_io_data_path_h5read(test_file, "testing");
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(pressio_float_dtype, pressio_data_dtype(result));
    ASSERT_EQ(3, pressio_data_num_dimensions(result));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 0));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 1));
    ASSERT_EQ(3, pressio_data_get_dimension(result, 2));
    std::vector<float> result_data(3*3*3);
    size_t size_in_bytes;
    void* result_ptr = pressio_data_ptr(result, &size_in_bytes);
    memcpy(result_data.data(), result_ptr, size_in_bytes);
    ASSERT_EQ(result_data.back(), 0.0f);
    EXPECT_EQ(floats, result_data);
    pressio_data_free(result);
  }


  pressio_data_free(data);
}

TEST_F(PressioIOHDFTests, partial_read_write) {
  std::vector<int32_t> ints(3*3*3);
  std::vector<size_t> dims{3,3,3};
  std::iota(std::begin(ints), std::end(ints), 0.0);

  //write the entire dataset
  pressio_data* data = pressio_data_new_nonowning(pressio_int32_dtype, ints.data(), dims.size(), dims.data());
  pressio_io_data_path_h5write(data, test_file, "partial");

  //read it back partially
  pressio library;
  auto io = library.get_io("hdf5");
  io->set_options({
    {"io:path", std::string(test_file)},
    {"hdf5:dataset", "partial"},
    {"hdf5:file_start", pressio_data{1,0,0}},
    {"hdf5:file_count", pressio_data{1,3,3}},
      });
  auto partial_data = io->read(nullptr);
  {
    ASSERT_NE(partial_data, nullptr) << io->error_msg();
    auto ptr = static_cast<int32_t*>(partial_data->data());
    size_t elms = partial_data->num_elements();
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(elms, 9);

    //make some changes to the middle slice and write it back partially
    std::transform(ptr, ptr+elms, ptr, [](int32_t e){ return e + 3; });
  }
  ASSERT_EQ(io->write(partial_data), 0);
  pressio_data_free(partial_data);

  //read back the entire dataset
  pressio_data* after_change = pressio_io_data_path_h5read(test_file, "partial");
  ASSERT_NE(after_change, nullptr);
  {
    auto ptr = static_cast<int32_t*>(after_change->data());
    auto nelms = after_change->num_elements();
    const std::vector<int32_t> expected = {  0, 1, 2, 3, 4, 5, 6, 7, 8,
                                            12,13,14,15,16,17,18,19,20,
                                            18,19,20,21,22,23,24,25,26,
                                    };
    const std::vector<int32_t> actual(ptr, ptr+nelms);

    EXPECT_EQ(actual, expected);
  }
}
