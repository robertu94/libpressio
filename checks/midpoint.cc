#include <numeric>
#include <iostream>

int main()
{
  float f1 = 1.0;
  float f2 = 3.0;
  float midpoint = std::midpoint(f1, f2);
  std::cout << midpoint << std::endl;
  return 0;
}
