#include <optional>

int main(int argc, char *argv[])
{
  std::optional<int> o = 3;
  return *o;
}
