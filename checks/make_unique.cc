#include <memory>
int main(int argc, char *argv[])
{
  auto i = std::make_unique<int>(3);
  return *i;
}
