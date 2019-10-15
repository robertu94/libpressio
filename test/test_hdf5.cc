#include <vector>
#include <numeric>
#include "libpressio_ext/io/hdf5.h"
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
  }

  std::iota(std::rbegin(floats), std::rend(floats), 0);


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
  }


  pressio_data_free(data);
}
