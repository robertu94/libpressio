#include <span>

int main()
{
  int v = 3;
  int* vs[] = {&v, &v, &v};
  const int* cvs[] = {&v, &v, &v};
  std::span<int*> s(vs, 3);
  std::span<const int*> cs(cvs, 3);
  std::span<const int* const> scic_from_s(s);
  std::span<const int* const> scic_from_cs(s);
  return 0;
}
