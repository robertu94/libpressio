#include "lossy_option.h"
#include "lossy_options.h"
#include "lossy_options_iter.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class LossyOptionsTests: public ::testing::Test {
  protected:
    void SetUp() {
      o = lossy_options_new();
      EXPECT_NE(o, nullptr);


      lossy_options_set_integer(o, "int", 1);
      lossy_options_set_double(o, "double", 1.2);
      lossy_options_set_string(o, "string", "testing");
      lossy_options_set_userptr(o, "data", &data);
    }

    void TearDown() {
      lossy_options_free(o);
    }

    struct lossy_options* o;
    struct lossy_option* op;
    struct user_data{
      int a;
      double b;
    } data = {1,2.0};
};

TEST_F(LossyOptionsTests, MakeOptions) {
  struct lossy_options* o = lossy_options_new();
  EXPECT_NE(o, nullptr);
  lossy_options_free(o);
}

TEST_F(LossyOptionsTests, TestInt) {

  op = lossy_options_get(o, "int");
  EXPECT_EQ(lossy_option_get_integer(op), 1);
  lossy_option_free(op);
}

TEST_F(LossyOptionsTests, TestDouble) {
  op = lossy_options_get(o, "double");
  EXPECT_EQ(lossy_option_get_double(op), 1.2);
  lossy_option_free(op);
}

TEST_F(LossyOptionsTests, TestString) {
  op = lossy_options_get(o, "string");
  EXPECT_THAT(lossy_option_get_string(op), ::testing::StrEq("testing"));
  lossy_option_free(op);
}

TEST_F(LossyOptionsTests, TestUserData) {
  op = lossy_options_get(o, "data");
  EXPECT_EQ((user_data*)lossy_option_get_userptr(op), &data);
  EXPECT_EQ(((user_data*)lossy_option_get_userptr(op))->a, 1);
  EXPECT_EQ(((user_data*)lossy_option_get_userptr(op))->b, 2.0);
  lossy_option_free(op);
}

TEST_F(LossyOptionsTests, IterateKeys) {
  int count = 0;
  auto it = lossy_options_get_iter(o);
  while(lossy_options_iter_has_value(it)) {
    const char* key = lossy_options_iter_get_key(it);
    struct lossy_option const* value = lossy_options_iter_get_value(it);
    switch(lossy_option_get_type(value)) {
      case lossy_option_charptr_type:
        EXPECT_THAT(key, ::testing::StrEq("string"));
        EXPECT_THAT(lossy_option_get_string(value), ::testing::StrEq("testing"));
        break;
      case lossy_option_double_type:
        EXPECT_THAT(key, ::testing::StrEq("double"));
        EXPECT_EQ(lossy_option_get_double(value), 1.2);
        break;
      case lossy_option_float_type:
        FAIL();
        break;
      case lossy_option_int32_type:
        EXPECT_THAT(key, ::testing::StrEq("int"));
        EXPECT_EQ(lossy_option_get_integer(value), 1);
        break;
      case lossy_option_uint32_type:
        FAIL();
        break;
      case lossy_option_userptr_type:
        EXPECT_THAT(key, ::testing::StrEq("data"));
        EXPECT_EQ(lossy_option_get_userptr(value), &data);
        break;
      case lossy_option_unset:
        FAIL();
    }

    count++;
    lossy_options_iter_next(it);
  }

  EXPECT_EQ(count, 4);
  lossy_options_iter_free(it);
}

