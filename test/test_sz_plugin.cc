#include <sz/sz.h>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "pressio.h"
#include "pressio_version.h"
#include "pressio_compressor.h"

class PressioCompressor: public ::testing::Test {
  protected:
    void SetUp() {
      library = pressio_instance();
      compressor = pressio_get_compressor(library, "sz");
    }

    void TearDown() {
      pressio_release(&library);
    }

    struct pressio* library;
    struct pressio_compressor* compressor;
};

TEST_F(PressioCompressor, VersionTest) {
  EXPECT_THAT(pressio_version(), ::testing::StrEq(LIBPRESSIO_VERSION));
  EXPECT_EQ(pressio_major_version(), LIBPRESSIO_MAJOR_VERSION);
  EXPECT_EQ(pressio_minor_version(), LIBPRESSIO_MINOR_VERSION);
  EXPECT_EQ(pressio_patch_version(), LIBPRESSIO_PATCH_VERSION);
  EXPECT_EQ(pressio_compressor_major_version(compressor), SZ_VER_MAJOR);
  EXPECT_EQ(pressio_compressor_minor_version(compressor), SZ_VER_MINOR);
  EXPECT_EQ(pressio_compressor_patch_version(compressor), SZ_VER_BUILD);
}

