#include <cstdlib>
#include "pressio_option.h"
#include "pressio_options.h"
#include "pressio_options_iter.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class PressioOptionsTests: public ::testing::Test {
  protected:
    void SetUp() {
      o = pressio_options_new();
      EXPECT_NE(o, nullptr);


      pressio_options_set_integer(o, "int", 1);
      pressio_options_set_double(o, "double", 1.2);
      pressio_options_set_string(o, "string", "testing");
      pressio_options_set_userptr(o, "data", &data);
    }

    void TearDown() {
      pressio_options_free(o);
    }

    struct pressio_options* o;
    struct pressio_option* op;
    struct user_data{
      int a;
      double b;
    } data = {1,2.0};
};

TEST_F(PressioOptionsTests, MakeOptions) {
  struct pressio_options* o = pressio_options_new();
  EXPECT_NE(o, nullptr);
  pressio_options_free(o);
}

TEST_F(PressioOptionsTests, TestInt) {

  op = pressio_options_get(o, "int");
  EXPECT_EQ(pressio_option_get_integer(op), 1);
  pressio_option_free(op);
}

TEST_F(PressioOptionsTests, TestDouble) {
  op = pressio_options_get(o, "double");
  EXPECT_EQ(pressio_option_get_double(op), 1.2);
  pressio_option_free(op);
}

TEST_F(PressioOptionsTests, TestString) {
  op = pressio_options_get(o, "string");
  EXPECT_THAT(pressio_option_get_string(op), ::testing::StrEq("testing"));
  pressio_option_free(op);
}

TEST_F(PressioOptionsTests, TestUserData) {
  op = pressio_options_get(o, "data");
  EXPECT_EQ((user_data*)pressio_option_get_userptr(op), &data);
  EXPECT_EQ(((user_data*)pressio_option_get_userptr(op))->a, 1);
  EXPECT_EQ(((user_data*)pressio_option_get_userptr(op))->b, 2.0);
  pressio_option_free(op);
}

TEST_F(PressioOptionsTests, IterateKeys) {
  int count = 0;
  auto it = pressio_options_get_iter(o);
  while(pressio_options_iter_has_value(it)) {
    const char* key = pressio_options_iter_get_key(it);
    struct pressio_option const* value = pressio_options_iter_get_value(it);
    switch(pressio_option_get_type(value)) {
      case pressio_option_charptr_type:
        EXPECT_THAT(key, ::testing::StrEq("string"));
        EXPECT_THAT(pressio_option_get_string(value), ::testing::StrEq("testing"));
        break;
      case pressio_option_double_type:
        EXPECT_THAT(key, ::testing::StrEq("double"));
        EXPECT_EQ(pressio_option_get_double(value), 1.2);
        break;
      case pressio_option_float_type:
        FAIL();
        break;
      case pressio_option_int32_type:
        EXPECT_THAT(key, ::testing::StrEq("int"));
        EXPECT_EQ(pressio_option_get_integer(value), 1);
        break;
      case pressio_option_uint32_type:
        FAIL();
        break;
      case pressio_option_userptr_type:
        EXPECT_THAT(key, ::testing::StrEq("data"));
        EXPECT_EQ(pressio_option_get_userptr(value), &data);
        break;
      case pressio_option_unset_type:
        FAIL();
        break;
    }

    count++;
    pressio_options_iter_next(it);
  }

  EXPECT_EQ(count, 4);
  pressio_options_iter_free(it);
}

TEST_F(PressioOptionsTests, Conversions ) {
  struct pressio_option* converted;

  //test implicit conversions
  op = pressio_options_get(o, "int");
  converted = pressio_option_convert_implicit(op, pressio_option_double_type);
  EXPECT_EQ(pressio_option_get_double(converted), 1.0);
  pressio_option_free(converted);
  pressio_option_free(op);

  //test explicit conversions fail when used with pressio_option_convert_implicit
  op = pressio_options_get(o, "double");
  converted = pressio_option_convert_implicit(op, pressio_option_int32_type);
  EXPECT_EQ(converted, nullptr);
  //no need to free converted, no memory allocated

  //test explicit conversion succeed
  converted = pressio_option_convert(op, pressio_option_int32_type, pressio_conversion_explicit);
  EXPECT_EQ(pressio_option_get_integer(converted), 1);
  pressio_option_free(converted);

  //check that it also works with special
  converted = pressio_option_convert(op, pressio_option_int32_type, pressio_conversion_special);
  EXPECT_EQ(pressio_option_get_integer(converted), 1);
  pressio_option_free(converted);

  pressio_option_free(op);
}

TEST_F(PressioOptionsTests, OptionConversions ) {
  struct pressio_option* converted;

  //test implicit conversions
  double d = 9.2;
  if(pressio_options_as_double(o, "int", &d) == pressio_options_key_set) {
    EXPECT_EQ(d, 1.0);
  } else {
    FAIL() << "conversion from int->double should have succeeded implicitly";
  }

  for (auto level : {pressio_conversion_implicit, pressio_conversion_explicit, pressio_conversion_special}) {
    d = 9.2;
    if(pressio_options_cast_double(o, "int", level, &d) == pressio_options_key_set) {
      EXPECT_EQ(d, 1.0);
    } else {
      FAIL() << "conversion int->double should have succeeded";
    }
    
  }

  int i = 3;
  if(pressio_options_as_integer(o, "double", &i) != pressio_options_key_exists) {
    FAIL() << "conversion from double->integer should fail implicit";
  }

  i = 3;
  if(pressio_options_cast_integer(o, "double", pressio_conversion_explicit, &i) == pressio_options_key_set) {
    EXPECT_EQ(i, 1);
  } else {
    FAIL() << "conversion from double->integer should succeed explicitly";
  }
}

TEST_F(PressioOptionsTests, StrConversions) {
  char* str;
  if(pressio_options_cast_string(o, "int", pressio_conversion_special, &str) == pressio_options_key_set)
  {
    EXPECT_THAT(str, ::testing::StrEq("1"));
  } else {
    FAIL() << "int should convert to string with special conversions";
  }
  free(str);

  if(pressio_options_cast_string(o, "double", pressio_conversion_special, &str) == pressio_options_key_set)
  {
    //implemented using std::to_string which has a default precision is 6 which it gets from std::sprintf
    EXPECT_THAT(str, ::testing::StrEq("1.200000"));
  } else {
    FAIL() << "int should convert to string with special conversions";
  }
  free(str);


}

TEST_F(PressioOptionsTests, SpecialConversions) {
  struct pressio_options* options = pressio_options_new();
  pressio_options_set_string(options, "numeric", "1.0");
  pressio_options_set_string(options, "numeric_bad", "asdf");
  struct pressio_option* numeric = pressio_options_get(options, "numeric");
  struct pressio_option* converted = pressio_option_convert(numeric, pressio_option_double_type, pressio_conversion_special);
  EXPECT_EQ(pressio_option_get_double(converted), 1.0);

  struct pressio_option* numeric_bad = pressio_options_get(options, "numeric_bad");
  struct pressio_option* bad_conversion = pressio_option_convert(numeric_bad, pressio_option_double_type, pressio_conversion_special);
  EXPECT_EQ(bad_conversion, nullptr);

  pressio_option_free(numeric);
  pressio_option_free(numeric_bad);
  pressio_option_free(converted);
  pressio_options_free(options);
}


TEST_F(PressioOptionsTests, OptionNewFreeFunction) {
  {
    struct pressio_option* option = pressio_option_new_integer(3);
    EXPECT_EQ(pressio_option_get_integer(option), 3);
    pressio_option_free(option);
  }

  {
    struct pressio_option* option = pressio_option_new();
    EXPECT_EQ(pressio_option_has_value(option), false);
    pressio_option_set_integer(option, 3);
    EXPECT_EQ(pressio_option_get_integer(option), 3);
    pressio_option_free(option);
  }

  
}


TEST_F(PressioOptionsTests, OptionSet) {

  struct pressio_option* option = pressio_option_new_integer(3);
  struct pressio_options* options = pressio_options_new();
  pressio_options_set(options, "foo", option);

  int result;
  if(pressio_options_get_integer(options, "foo", &result) == pressio_options_key_set) {
    EXPECT_EQ(result, 3);
  } else {
    FAIL();
  }
  

  pressio_option_free(option);
  pressio_options_free(options);
}


TEST_F(PressioOptionsTests, OptionAssign) {

  struct pressio_option* lhs = pressio_option_new_integer(3);
  struct pressio_option* rhs = pressio_option_new_double(4.0);

  if(pressio_option_as_set(lhs, rhs) == pressio_options_key_set) {
    FAIL() << "double should not be implicltly assignable to int";
  } 

  if(pressio_option_cast_set(lhs, rhs, pressio_conversion_explicit) == pressio_options_key_set) {
    EXPECT_EQ(pressio_option_get_integer(lhs), 4);
  } 
  pressio_option_free(lhs);
  pressio_option_free(rhs);
}


TEST_F(PressioOptionsTests, OptionsAssign) {

  struct pressio_options* options = pressio_options_new();
  pressio_options_set_integer(options, "int", 3);



  struct pressio_option* rhs = pressio_option_new_double(4.0);

  if(pressio_options_as_set(options, "int", rhs) == pressio_options_key_set) {
    FAIL() << "double should not be implicltly assignable to int";
  } 

  if(pressio_options_cast_set(options, "int", rhs, pressio_conversion_explicit) == pressio_options_key_set) {
    int result;
    if(pressio_options_get_integer(options, "int", &result) == pressio_options_key_set) {
      EXPECT_EQ(result, 4);
    } else {
      FAIL() << "failed to set integer";
    }
  } 
}
