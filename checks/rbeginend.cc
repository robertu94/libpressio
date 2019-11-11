#include <vector>
#include <iterator>

int main(int argc, char *argv[])
{
  int sum;
  std::vector<int> foo{1,2,2};
  for (auto it = std::rbegin(foo); it != std::rend(foo); ++it) {
    sum+=*it;
  }
  return sum;
}
