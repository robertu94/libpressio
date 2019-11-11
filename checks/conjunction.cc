#include <type_traits>
int main(int argc, char *argv[])
{
  
  return std::conjunction<std::is_same<int,int>, std::is_same<float,float>>::value;
}
