#include <vector>
#include <numeric>
#include "lossy_data.h"
#include "gtest/gtest.h"

class LossyDataTests: public ::testing::Test {
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

TEST_F(LossyDataTests, MakeLossyData) {
  lossy_data* d = lossy_data_new_nonowning(lossy_int32_dtype, data.data(), 2, dims);
  EXPECT_NE(d, nullptr);
  EXPECT_EQ(lossy_data_dtype(d), lossy_int32_dtype);
  EXPECT_EQ(lossy_data_num_dimentions(d), 2);
  EXPECT_EQ(lossy_data_get_dimention(d, 0), 2);
  EXPECT_EQ(lossy_data_get_dimention(d, 1), 3);
  EXPECT_EQ(lossy_data_ptr(d, nullptr), data.data());
  size_t size = 0;
  EXPECT_EQ(lossy_data_ptr(d, &size), data.data());
  EXPECT_EQ(size, 6*sizeof(int));

  lossy_data_free(d);
}


TEST_F(LossyDataTests, MakeCopy) {
  lossy_data* d = lossy_data_new_nonowning(lossy_int32_dtype, data.data(), 2, dims);
  EXPECT_NE(d, nullptr);
  size_t size = 0;
  void* copy = lossy_data_copy(d, &size);
  void* copy2 = lossy_data_copy(d, &size);
  EXPECT_NE(copy, data.data());
  EXPECT_NE(copy, copy2);

  lossy_data_free(d);
  free(copy);
  free(copy2);
}
