#include <variant>

int main(int argc, char *argv[])
{
  std::variant<std::monostate, int, float> v;
  v = 1.2f;
  v = 1;
  return std::get<int>(v);
}
