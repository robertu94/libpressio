#include <type_traits>
int main()
{
  return std::negation<std::false_type>::value;
}
