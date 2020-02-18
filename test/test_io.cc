#include <vector>
#include <numeric>
#include "libpressio_ext/io/pressio_io.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
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
    pressio library{};
};

TEST_F(PressioDataIOTests, TestReadNullptr) {
  auto data = pressio_io_data_read(nullptr, tmp_fd);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_byte_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestReadNullptrGeneric) {
  auto io = pressio_get_io(&library, "posix");
  (*io)->set_options({
      {"io:file_descriptor", tmp_fd}
  });
  auto data = pressio_io_read(io, nullptr);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_byte_dtype);
  pressio_data_free(data);
  pressio_io_free(io);
}

TEST_F(PressioDataIOTests, TestReadEmptyGeneric) {
  size_t sizes[] = {2,3};
  auto io = pressio_get_io(&library, "posix");
  (*io)->set_options({
      {"io:file_descriptor", tmp_fd}
  });
  auto size_info = pressio_data_new_empty(pressio_int32_dtype, 2, sizes);
  auto data = pressio_io_read(io, size_info);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
  pressio_io_free(io);
}


TEST_F(PressioDataIOTests, TestReadEmpty) {
  size_t sizes[] = {2,3};
  auto size_info = pressio_data_new_empty(pressio_int32_dtype, 2, sizes);
  auto data = pressio_io_data_read(size_info, tmp_fd);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestReadFullGeneric) {
  size_t sizes[] = {2,3};
  auto buffer = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  auto buffer_ptr = pressio_data_ptr(buffer, nullptr);
  auto io = pressio_get_io(&library, "posix");
  (*io)->set_options({
      {"io:file_descriptor", tmp_fd}
  });
  auto data = pressio_io_read(io, buffer);
  ASSERT_NE(data, nullptr);
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
  pressio_io_free(io);
}

TEST_F(PressioDataIOTests, TestReadFull) {
  size_t sizes[] = {2,3};
  auto buffer = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  auto buffer_ptr = pressio_data_ptr(buffer, nullptr);
  auto data = pressio_io_data_read(buffer, tmp_fd);
  ASSERT_NE(data, nullptr);
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}


TEST_F(PressioDataIOTests, TestReadPathNullptrGeneric) {
  auto io = pressio_get_io(&library, "posix");
  (*io)->set_options({
      {"io:path", tmp_name}
  });
  auto data = pressio_io_read(io, nullptr);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_byte_dtype);
  pressio_data_free(data);
  pressio_io_free(io);
}

TEST_F(PressioDataIOTests, TestReadPathNullptr) {
  auto data = pressio_io_data_path_read(nullptr, tmp_name.c_str());
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_byte_dtype);
  pressio_data_free(data);
}


TEST_F(PressioDataIOTests, TestReadPathEmptyGeneric) {
  size_t sizes[] = {2,3};
  auto size_info = pressio_data_new_empty(pressio_int32_dtype, 2, sizes);
  auto io = pressio_get_io(&library, "posix");
  (*io)->set_options({
      {"io:path", tmp_name}
  });
  auto data = pressio_io_read(io, size_info);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
  pressio_io_free(io);
}
TEST_F(PressioDataIOTests, TestReadPathEmpty) {
  size_t sizes[] = {2,3};
  auto size_info = pressio_data_new_empty(pressio_int32_dtype, 2, sizes);
  auto data = pressio_io_data_path_read(size_info, tmp_name.c_str());
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}


TEST_F(PressioDataIOTests, TestReadPathFullGeneric) {
  size_t sizes[] = {2,3};
  auto buffer = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  auto buffer_ptr = pressio_data_ptr(buffer, nullptr);
  auto io = pressio_get_io(&library, "posix");
  (*io)->set_options({
      {"io:path", tmp_name}
  });
  auto data = pressio_io_read(io, buffer);
  ASSERT_NE(data, nullptr);
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
  pressio_io_free(io);
}
TEST_F(PressioDataIOTests, TestReadPathFull) {
  size_t sizes[] = {2,3};
  auto buffer = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  auto buffer_ptr = pressio_data_ptr(buffer, nullptr);
  auto data = pressio_io_data_path_read(buffer, tmp_name.c_str());
  ASSERT_NE(data, nullptr);
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
}

TEST_F(PressioDataIOTests, TestFReadNullptr) {
  auto tmp = fdopen(tmp_fd, "r");
  auto data = pressio_io_data_fread(nullptr, tmp);
  ASSERT_NE(data, nullptr);
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
  ASSERT_NE(data, nullptr);
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
  ASSERT_NE(data, nullptr);
  auto data_ptr = pressio_data_ptr(data, nullptr);
  EXPECT_EQ(buffer_ptr, data_ptr);
  EXPECT_EQ(pressio_data_get_bytes(data), sizeof(int)*6);
  EXPECT_EQ(pressio_data_dtype(data), pressio_int32_dtype);
  pressio_data_free(data);
  fclose(tmp);
}


TEST_F(PressioDataIOTests, TestWriteGeneric) {
  size_t sizes[] = {2,3};
  auto data = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  size_t buffer_size;
  int* buffer = static_cast<int*>(pressio_data_ptr(data, &buffer_size));
  std::iota(buffer, buffer + pressio_data_num_elements(data), 0);
  auto tmpwrite_name = std::string("test_io_readXXXXXX");
  auto tmpwrite_fd = mkstemp(const_cast<char*>(tmpwrite_name.data()));
  auto io = pressio_get_io(&library, "posix");
  (*io)->set_options({
      {"io:file_descriptor", tmpwrite_fd}
  });
  EXPECT_EQ(pressio_io_write(io, data), 0);
  pressio_data_free(data);
  close(tmpwrite_fd);
  unlink(tmpwrite_name.data());
}


TEST_F(PressioDataIOTests, TestWrite) {
  size_t sizes[] = {2,3};
  auto data = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  size_t buffer_size;
  int* buffer = static_cast<int*>(pressio_data_ptr(data, &buffer_size));
  std::iota(buffer, buffer + pressio_data_num_elements(data), 0);
  auto tmpwrite_name = std::string("test_io_readXXXXXX");
  auto tmpwrite_fd = mkstemp(const_cast<char*>(tmpwrite_name.data()));
  EXPECT_EQ(pressio_io_data_write(data, tmpwrite_fd), sizeof(int)*6);
  pressio_data_free(data);
  close(tmpwrite_fd);
  unlink(tmpwrite_name.data());
}

TEST_F(PressioDataIOTests, TestWriteCSV) {
  size_t sizes[] = {2,3};
  auto data = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  size_t buffer_size;
  int* buffer = static_cast<int*>(pressio_data_ptr(data, &buffer_size));
  std::iota(buffer, buffer + pressio_data_num_elements(data), 0);

  auto tmpwrite_name = std::string("test_io_readXXXXXX");
  auto tmpwrite_fd = mkstemp(const_cast<char*>(tmpwrite_name.data()));

  auto io = pressio_get_io(&library, "csv");
  (*io)->set_options({
      {"io:path", tmpwrite_name},
      {"csv:headers", std::vector<std::string>{"foo", "bar", "sue"}}
  });
  EXPECT_EQ(pressio_io_write(io, data), 0);
  pressio_data_free(data);
  close(tmpwrite_fd);
  unlink(tmpwrite_name.data());
}


TEST_F(PressioDataIOTests, TestReadCSV) {
  size_t sizes[] = {2,3};
  auto data = pressio_data_new_owning(pressio_int32_dtype, 2, sizes);
  size_t buffer_size;
  int* buffer = static_cast<int*>(pressio_data_ptr(data, &buffer_size));
  std::iota(buffer, buffer + pressio_data_num_elements(data), 0);

  auto tmpwrite_name = std::string("test_io_readXXXXXX");
  auto tmpwrite_fd = mkstemp(const_cast<char*>(tmpwrite_name.data()));

  auto io = pressio_get_io(&library, "csv");
  (*io)->set_options({
      {"io:path", tmpwrite_name},
      {"csv:headers", std::vector<std::string>{"foo", "bar", "sue"}},
      {"csv:skip_rows", 1u}
  });
  EXPECT_EQ(pressio_io_write(io, data), 0);
  pressio_data* read = pressio_io_read(io, nullptr);

  EXPECT_EQ(pressio_data_num_dimensions(read), 2);
  EXPECT_EQ(pressio_data_get_dimension(read, 0), 2);
  EXPECT_EQ(pressio_data_get_dimension(read, 1), 3);

  pressio_data_free(data);
  pressio_data_free(read);
  close(tmpwrite_fd);
  unlink(tmpwrite_name.data());
}
