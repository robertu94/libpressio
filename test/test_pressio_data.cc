#include <vector>
#include <numeric>
#include <memory>
#include <array>
#include "pressio_data.h"
#include "libpressio_ext/cpp/data.h"
#include "multi_dimensional_iterator.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class PressioDataTests: public ::testing::Test {
  protected:
    void SetUp() {
      data = std::vector<int>(6);
      std::iota(std::begin(data), std::end(data), 0);
    }

    void TearDown() {
    }

    const size_t dims[2]  = {2,3};
    std::vector<int> data;
};

TEST_F(PressioDataTests, MakePressioData) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  EXPECT_NE(d, nullptr);
  EXPECT_EQ(pressio_data_dtype(d), pressio_int32_dtype);
  EXPECT_EQ(pressio_data_num_dimensions(d), 2);
  EXPECT_EQ(pressio_data_get_dimension(d, 0), 2);
  EXPECT_EQ(pressio_data_get_dimension(d, 1), 3);
  EXPECT_EQ(pressio_data_ptr(d, nullptr), data.data());
  size_t size = 0;
  EXPECT_EQ(pressio_data_ptr(d, &size), data.data());
  EXPECT_EQ(size, 6*sizeof(int));

  pressio_data_free(d);
}

namespace {
struct accumulator {
  template <class T>
    double operator()(T begin, T end) {
        return static_cast<double>(std::accumulate(begin, end, 0));
    }
};
}

TEST_F(PressioDataTests, ForEach) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  auto result = pressio_data_for_each<double>(*d, accumulator{});

  EXPECT_EQ(result, static_cast<double>(std::accumulate(std::begin(data), std::end(data), 0)));
  pressio_data_free(d);
}



TEST_F(PressioDataTests, MakeCopy) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  EXPECT_NE(d, nullptr);
  size_t size = 0;
  void* copy = pressio_data_copy(d, &size);
  void* copy2 = pressio_data_copy(d, &size);
  EXPECT_NE(copy, data.data());
  EXPECT_NE(copy, copy2);

  pressio_data_free(d);
  free(copy);
  free(copy2);
}
TEST_F(PressioDataTests, Clone) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  pressio_data* cloned = pressio_data_new_clone(d);
  EXPECT_NE(d, nullptr);
  size_t bytes, bytes2;
  unsigned char * orig = (unsigned char*)pressio_data_ptr(d, &bytes);
  unsigned char * clone = (unsigned char*)pressio_data_ptr(cloned, &bytes2);
  ASSERT_EQ(bytes, bytes2);
  ASSERT_NE(orig, clone);
  EXPECT_TRUE(memcpy(orig, clone, bytes));

  pressio_data_free(d);
  pressio_data_free(cloned);
}

TEST_F(PressioDataTests, Select) {
  size_t dim[] = {9ul, 10ul};
  auto* data = pressio_data_new_owning(pressio_int32_dtype, 2ul, dim);
  auto* ptr = static_cast<int*>(pressio_data_ptr(data, nullptr));
  std::iota(ptr, ptr+90, 0);


  std::array<size_t,2> start{1,0};
  std::array<size_t,2> stride{5,7};
  std::array<size_t,2> count{2,2};
  std::array<size_t,2> block{2,3};
  std::array<int,24> expected = {
     1,  2,    6,  7,
    10, 11,   15, 16,
    19, 20,   24, 25,

    64, 65,   69, 70,
    73, 74,   78, 79,
    82, 83,   87, 88,
  };

  auto* slab = pressio_data_select(data,
      start.data(),
      stride.data(),
      count.data(),
      block.data());

  ASSERT_EQ(pressio_data_num_elements(slab), 24);
  ASSERT_EQ(pressio_data_dtype(slab), pressio_int32_dtype);

  //check the actual values
  std::array<int,24> results;
  std::copy_n((int*)pressio_data_ptr(slab, nullptr), 24, begin(results));
  EXPECT_EQ(results, expected);

  pressio_data_free(data);
  pressio_data_free(slab);
}

TEST(test_mulit_dimensional_array, test_mulit_dimensional_array) {
  std::array<int, 12> values;
  std::iota(begin(values), end(values), 0);
  std::vector<size_t> global_dims = {3, 4};
  std::vector<size_t> stride = {2, 3};
  std::vector<size_t> count = {2, 2};
  std::vector<size_t> start = {0, 0};
  auto range = std::make_shared<multi_dimensional_range<int>>(
      values.data(),
      std::begin(global_dims),
      std::end(global_dims),
      std::begin(count),
      std::begin(stride),
      std::begin(start)
      );
  std::vector<int> results;
  int expected[] = {0, 2, 9, 11};
  std::copy(std::begin(*range), std::end(*range), std::back_inserter(results));
  EXPECT_THAT(results, ::testing::ElementsAreArray(expected));
}

TEST(test_mulit_dimensional_array, test_mulit_dimensional_array_default_start) {
  std::array<int, 12> values;
  std::iota(begin(values), end(values), 0);
  std::vector<size_t> global_dims = {3, 4};
  std::vector<size_t> stride = {2, 3};
  std::vector<size_t> count = {2, 2};
  auto range = std::make_shared<multi_dimensional_range<int>>(
      values.data(),
      std::begin(global_dims),
      std::end(global_dims),
      std::begin(count),
      std::begin(stride)
      );
  std::vector<int> results;
  int expected[] = {0, 2, 9, 11};
  std::copy(std::begin(*range), std::end(*range), std::back_inserter(results));
  EXPECT_THAT(results, ::testing::ElementsAreArray(expected));
}


TEST(test_mulit_dimensional_array, test_mulit_dimensional_array_default_stride) {
  std::array<int, 12> values;
  std::iota(begin(values), end(values), 0);
  std::vector<size_t> global_dims = {3, 4};
  std::vector<size_t> count = {2, 2};
  auto range = std::make_shared<multi_dimensional_range<int>>(
      values.data(),
      std::begin(global_dims),
      std::end(global_dims),
      std::begin(count)
      );
  std::vector<int> results;
  int expected[] = {0, 1, 3, 4};
  std::copy(std::begin(*range), std::end(*range), std::back_inserter(results));
  EXPECT_THAT(results, ::testing::ElementsAreArray(expected));
}

TEST(test_mulit_dimensional_array, test_mulit_dimensional_array_operator) {
  size_t dim[] = {9ul, 10ul};
  auto* data = pressio_data_new_owning(pressio_int32_dtype, 2ul, dim);
  auto* ptr = static_cast<int*>(pressio_data_ptr(data, nullptr));
  std::iota(ptr, ptr+90, 0);
  
  std::array<size_t,2> start{1,0};
  std::array<size_t,2> stride{5,7};
  std::array<size_t,2> count{2,2};
  auto range = std::make_shared<multi_dimensional_range<int>>(ptr,
      dim,
      dim+2,
      std::begin(count),
      std::begin(stride),
      std::begin(start)
      );
  EXPECT_EQ((*range)(0,0), 1);
  EXPECT_EQ((*range)(1,0), 6);
  EXPECT_EQ((*range)(0,1), 64);
  EXPECT_EQ((*range)(1,1), 69);
  ASSERT_EQ(range->num_dims(), 2);
  EXPECT_EQ(range->get_global_dims(0), 9);
  EXPECT_EQ(range->get_global_dims(1), 10);
  EXPECT_EQ(range->get_local_dims(0), 2);
  EXPECT_EQ(range->get_local_dims(1), 2);

  pressio_data_free(data);
}

