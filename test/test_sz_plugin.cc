#include <sz.h>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "lossy.h"
#include "lossy_version.h"
#include "lossy_compressor.h"

class LossyCompressor: public ::testing::Test {
  protected:
    void SetUp() {
      library = lossy_instance();
      compressor = lossy_get_compressor(library, "sz");
    }

    void TearDown() {
      lossy_release(&library);
    }

    struct lossy* library;
    struct lossy_compressor* compressor;
};

TEST_F(LossyCompressor, VersionTest) {
  EXPECT_THAT(lossy_version(), ::testing::StrEq(LIBLOSSY_VERSION));
  EXPECT_EQ(lossy_major_version(), LIBLOSSY_MAJOR_VERSION);
  EXPECT_EQ(lossy_minor_version(), LIBLOSSY_MINOR_VERSION);
  EXPECT_EQ(lossy_patch_version(), LIBLOSSY_PATCH_VERSION);
  EXPECT_EQ(lossy_compressor_major_version(compressor), SZ_VER_MAJOR);
  EXPECT_EQ(lossy_compressor_minor_version(compressor), SZ_VER_MINOR);
  EXPECT_EQ(lossy_compressor_patch_version(compressor), SZ_VER_BUILD);
}

