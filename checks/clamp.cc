#include <algorithm>

int main(int argc, char *[])
{
  return std::clamp(argc, 1, 3);
}
