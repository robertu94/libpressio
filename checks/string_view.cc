#include <string_view>
#include <string>

int main()
{
  const std::string_view v{"testing"};
  std::string s = std::string(v);
  
  return !v.empty();
}
