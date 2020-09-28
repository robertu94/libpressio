#include <type_traits>

int main()
{
  return std::is_null_pointer<std::nullptr_t>::value;
}
