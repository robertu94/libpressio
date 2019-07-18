#include <cstdlib>
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

TEST_F(LossyOptionsTests, Conversions ) {
  struct lossy_option* converted;

  //test implicit conversions
  op = lossy_options_get(o, "int");
  converted = lossy_option_convert_implicit(op, lossy_option_double_type);
  EXPECT_EQ(lossy_option_get_double(converted), 1.0);
  lossy_option_free(converted);
  lossy_option_free(op);

  //test explicit conversions fail when used with lossy_option_convert_implicit
  op = lossy_options_get(o, "double");
  converted = lossy_option_convert_implicit(op, lossy_option_int32_type);
  EXPECT_EQ(converted, nullptr);
  //no need to free converted, no memory allocated

  //test explicit conversion succeed
  converted = lossy_option_convert(op, lossy_option_int32_type, lossy_conversion_explicit);
  EXPECT_EQ(lossy_option_get_integer(converted), 1);
  lossy_option_free(converted);

  //check that it also works with special
  converted = lossy_option_convert(op, lossy_option_int32_type, lossy_conversion_special);
  EXPECT_EQ(lossy_option_get_integer(converted), 1);
  lossy_option_free(converted);

  lossy_option_free(op);
}

TEST_F(LossyOptionsTests, OptionConversions ) {
  struct lossy_option* converted;

  //test implicit conversions
  double d = 9.2;
  if(lossy_options_as_double(o, "int", &d) == lossy_options_key_set) {
    EXPECT_EQ(d, 1.0);
  } else {
    FAIL() << "conversion from int->double should have succeeded implicitly";
  }

  for (auto level : {lossy_conversion_implicit, lossy_conversion_explicit, lossy_conversion_special}) {
    d = 9.2;
    if(lossy_options_cast_double(o, "int", level, &d) == lossy_options_key_set) {
      EXPECT_EQ(d, 1.0);
    } else {
      FAIL() << "conversion int->double should have succeeded";
    }
    
  }

  int i = 3;
  if(lossy_options_as_integer(o, "double", &i) != lossy_options_key_exists) {
    FAIL() << "conversion from double->integer should fail implicit";
  }

  i = 3;
  if(lossy_options_cast_integer(o, "double", lossy_conversion_explicit, &i) == lossy_options_key_set) {
    EXPECT_EQ(i, 1);
  } else {
    FAIL() << "conversion from double->integer should succeed explicitly";
  }
}

TEST_F(LossyOptionsTests, StrConversions) {
  char* str;
  if(lossy_options_cast_string(o, "int", lossy_conversion_special, &str) == lossy_options_key_set)
  {
    EXPECT_THAT(str, ::testing::StrEq("1"));
  } else {
    FAIL() << "int should convert to string with special conversions";
  }
  free(str);

  if(lossy_options_cast_string(o, "double", lossy_conversion_special, &str) == lossy_options_key_set)
  {
    //implemented using std::to_string which has a default precision is 6 which it gets from std::sprintf
    EXPECT_THAT(str, ::testing::StrEq("1.200000"));
  } else {
    FAIL() << "int should convert to string with special conversions";
  }
  free(str);


}

TEST_F(LossyOptionsTests, SpecialConversions) {
  struct lossy_options* options = lossy_options_new();
  lossy_options_set_string(options, "numeric", "1.0");
  lossy_options_set_string(options, "numeric_bad", "asdf");
  struct lossy_option* numeric = lossy_options_get(options, "numeric");
  struct lossy_option* converted = lossy_option_convert(numeric, lossy_option_double_type, lossy_conversion_special);
  EXPECT_EQ(lossy_option_get_double(converted), 1.0);

  struct lossy_option* numeric_bad = lossy_options_get(options, "numeric_bad");
  struct lossy_option* bad_conversion = lossy_option_convert(numeric_bad, lossy_option_double_type, lossy_conversion_special);
  EXPECT_EQ(bad_conversion, nullptr);

  lossy_option_free(numeric);
  lossy_option_free(numeric_bad);
  lossy_option_free(converted);
  lossy_options_free(options);
}


TEST_F(LossyOptionsTests, OptionNewFreeFunction) {
  {
    struct lossy_option* option = lossy_option_new_integer(3);
    EXPECT_EQ(lossy_option_get_integer(option), 3);
    lossy_option_free(option);
  }

  {
    struct lossy_option* option = lossy_option_new();
    EXPECT_EQ(lossy_option_get_type(option), lossy_option_unset);
    lossy_option_set_integer(option, 3);
    EXPECT_EQ(lossy_option_get_integer(option), 3);
    lossy_option_free(option);
  }

  
}

