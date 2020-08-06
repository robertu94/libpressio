#include <algorithm>
#include <vector>
#include <cstdio>

int main()
{
  std::vector<int> v {5, 2, 4, 1, 3}; 
  std::nth_element(v.begin(), std::next(v.begin(), v.size()/2), v.end());
  std::printf("%d\n", v[v.size()/2]);
  return 0;
}
