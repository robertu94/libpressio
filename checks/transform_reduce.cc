#include <vector>
#include <numeric>

int main() {
  std::vector<int> v1 = {1,2,3};
  std::vector<int> v2 = {1,2,3};
  return std::transform_reduce(
      v1.begin(), v1.end(),
      v2.begin(),
      0,
      [](int a, int b) { return a+b; },
      [](int a, int b) { return a*b; }
      );
}
