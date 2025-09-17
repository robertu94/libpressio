#ifndef BASIC_INDEXER_H_E81HP9FW
#define BASIC_INDEXER_H_E81HP9FW
#include <std_compat/type_traits.h>
#include <std_compat/iterator.h>
#include <std_compat/functional.h>
#include <algorithm>
#include <numeric>
#include <array>

namespace libpressio { namespace utilities {

//assume dims are fastest to slowest
template <class SizeType, size_t N>
struct basic_indexer {
  template <class... T, typename std::enable_if<compat::conjunction<std::is_integral<typename std::decay<T>::type>...>::value,int>::type = 0>
  basic_indexer(T&&... args) noexcept: max_dims{static_cast<SizeType>(args)...} {}

  basic_indexer(std::array<SizeType,N> args) noexcept: max_dims(args) {}

  basic_indexer(std::initializer_list<SizeType> args) noexcept: max_dims([](std::initializer_list<SizeType> args){
        std::array<SizeType,N> dims;
        std::copy(args.begin(), args.end(), dims.begin());
        return dims;
      }(args)) {}

  template <class It>
  basic_indexer(It first, It second) noexcept:
    max_dims([](It first, It second){
        std::array<SizeType,N> dims;
        std::copy(first, second, dims.begin());
        return dims;
      }(first, second)) {
    }

  template <class... T>
  typename std::enable_if<compat::conjunction<std::is_integral<typename std::decay<T>::type>...>::value && sizeof...(T) >= 1,std::size_t>::type
  operator()(T&&... args) const noexcept {
    std::array<SizeType, sizeof...(T)> dims{static_cast<SizeType>(args)...};
    return operator()(dims);
  }
  SizeType operator()(std::array<SizeType, N> const idxs) const noexcept {
    SizeType idx = idxs.back();
    SizeType i = N-1;
    do  {
      i--;
      idx*= max_dims[i]; 
      idx+= idxs[i];
    } while (i);
    return idx;
  }

  SizeType operator[](std::size_t i) const noexcept {
    return max_dims[i]; }
  SizeType size() const noexcept {
    return std::accumulate(max_dims.begin(), max_dims.end(), SizeType{1}, compat::multiplies<>{});
  }

  std::vector<SizeType> as_vec() {
    return std::vector<SizeType>(max_dims.begin(), max_dims.end());
  }

  std::array<SizeType, N> const max_dims;
};

template <size_t N>
using indexer = basic_indexer<size_t, N>;

template <size_t N>
using sindexer = basic_indexer<ssize_t, N>;

template <class T, template <class,  size_t> class array_type, class V, size_t N>
array_type<T, N> as(array_type<V,N> const& in) {
  std::array<T,N> arr;
  for (size_t i = 0; i < N; ++i) {
    arr[i] = static_cast<T>(in[i]);
  }
  return arr;
}
} }
#endif /* end of include guard: BASIC_INDEXER_H_E81HP9FW */
