#include <utility>
int main() {
  int i = 1;
  int j = 2;
  j = std::exchange(i,0);
}
