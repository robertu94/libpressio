#include <vector>
#include <numeric>
#include "libpressio_ext/io/posix.h"
#include "pressio_data.h"
#include "gtest/gtest.h"

class PressioDataIOTests: public ::testing::Test {
  protected:
    void SetUp() {
      data = std::vector<int>(size);
      std::iota(std::begin(data), std::end(data), 0);
      tmp_name = std::string("test_io_readXXXXXX\0");
      tmp_fd = mkstemp(const_cast<char*>(tmp_name.data()));
      size_t bytes_written = write(tmp_fd, data.data(), sizeof(int)*data.size());
      ASSERT_EQ(bytes_written, data.size() * sizeof(int));
      lseek(tmp_fd, 0, SEEK_SET);
    }

    void TearDown() {
      close(tmp_fd);
      unlink(tmp_name.data());
    }

    const size_t dims[2]  = {2,3};
    std::vector<int> data;
    int tmp_fd;
    std::string tmp_name;
    const size_t size = 6;
};

TEST_F(PressioDataIOTests, TestReadNullptr) {
  auto data = pressio_io_data_read(nullptr, tmp_fd);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_byte_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestReadEmpty) {
  size_t sizes[] = {2,3};
  auto size_info = pressio_data_new_empty(pressio_int32_dtype, 2, sizes);
  auto data = pressio_io_data_read(size_info, tmp_fd);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestReadFull) {
  size_t sizes[] = {2,3};
  auto buffer = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  auto buffer_ptr = pressio_data_ptr(buffer, nullptr);
  auto data = pressio_io_data_read(buffer, tmp_fd);
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}


TEST_F(PressioDataIOTests, TestReadPathNullptr) {
  auto data = pressio_io_data_path_read(nullptr, tmp_name.c_str());
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_byte_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestReadPathEmpty) {
  size_t sizes[] = {2,3};
  auto size_info = pressio_data_new_empty(pressio_int32_dtype, 2, sizes);
  auto data = pressio_io_data_path_read(size_info, tmp_name.c_str());
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestReadPathFull) {
  size_t sizes[] = {2,3};
  auto buffer = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  auto buffer_ptr = pressio_data_ptr(buffer, nullptr);
  auto data = pressio_io_data_path_read(buffer, tmp_name.c_str());
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestFReadNullptr) {
  auto tmp = fdopen(tmp_fd, "r");
  auto data = pressio_io_data_fread(nullptr, tmp);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_byte_dtype);
  pressio_data_free(data);
  fclose(tmp);
}

TEST_F(PressioDataIOTests, TestFReadEmpty) {
  size_t sizes[] = {2,3};
  auto size_info = pressio_data_new_empty(pressio_int32_dtype, 2, sizes);
  auto tmp = fdopen(tmp_fd, "r");
  auto data = pressio_io_data_fread(size_info, tmp);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
  fclose(tmp);
}

TEST_F(PressioDataIOTests, TestFReadFull) {
  size_t sizes[] = {2,3};
  auto buffer = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  auto buffer_ptr = pressio_data_ptr(buffer, nullptr);
  auto tmp = fdopen(tmp_fd, "r");
  auto data = pressio_io_data_fread(buffer, tmp);
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
  fclose(tmp);
}



TEST_F(PressioDataIOTests, TestWrite) {
  size_t sizes[] = {2,3};
  auto data = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  size_t buffer_size;
  int* buffer = static_cast<int*>(pressio_data_ptr(data, &buffer_size));
  std::iota(buffer, buffer + pressio_data_num_dimensions(data), 0);
  auto tmpwrite_name = std::string("test_io_readXXXXXX\0");
  auto tmpwrite_fd = mkstemp(const_cast<char*>(tmpwrite_name.data()));
  EXPECT_EQ(pressio_io_data_write(data, tmpwrite_fd), sizeof(int)*6);
  pressio_data_free(data);
  close(tmpwrite_fd);
  unlink(tmpwrite_name.data());
}
