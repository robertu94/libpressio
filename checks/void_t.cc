#include <type_traits>
int main()
{
  return std::is_same<std::void_t<int, float>,void>::value;
}
