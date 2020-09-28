/**
 * \file
 * \brief back ports of `<numeric>`
 */
#ifndef LIBPRESSIO_COMPAT_NUMERIC_H
#define LIBPRESSIO_COMPAT_NUMERIC_H
#include <pressio_version.h>
#include <limits>
#include <cstddef>
#include <numeric>
#include "functional.h"
#include "type_traits.h"

namespace compat {
#if !(LIBPRESSIO_COMPAT_HAS_MIDPOINT)

  /**
   * \returns the midpoint of two points
   * \param a the first point
   * \param b the second point
   */
template <class Type>
constexpr typename std::enable_if<std::is_integral<Type>::value &&
                                    !std::is_same<bool, Type>::value &&
                                    !compat::is_null_pointer<Type>::value,
                                  Type>::type
midpoint(Type a, Type b) noexcept
{
  using Up = typename std::make_unsigned<Type>::type;
  constexpr Up bitshift = std::numeric_limits<Up>::digits - 1;

  Up diff = Up(b) - Up(a);
  Up sign_bit = b < a;

  Up half_diff = (diff / 2) + (sign_bit << bitshift) + (sign_bit & diff);

  return a + half_diff;
}

  /**
   * \returns the midpoint of two points
   * \param a the first point
   * \param b the second point
   */
template <class TypePtr>
constexpr typename std::enable_if<
  std::is_pointer<TypePtr>::value &&
    std::is_object<typename std::remove_pointer<TypePtr>::type>::value &&
    !std::is_void<typename std::remove_pointer<TypePtr>::type>::value &&
    (sizeof(typename std::remove_pointer<TypePtr>::type) > 0),
  TypePtr>::type
midpoint(TypePtr a, TypePtr b) noexcept
{
  return a + compat::midpoint(ptrdiff_t(0), b - a);
}

namespace impl {
  /**
   * \param f the input
   * returns the absolute value of f
   */
  template <typename Floating> constexpr Floating fp_abs(Floating f) { return f >= 0 ? f : -f; }
} // namespace impl

  /**
   * \returns the midpoint of two points
   * \param a the first point
   * \param b the second point
   */
template <class Floating>
constexpr typename std::enable_if<std::is_floating_point<Floating>::value,
                                  Floating>::type
midpoint(Floating a, Floating b) noexcept
{
  constexpr Floating low = std::numeric_limits<Floating>::min() * 2;
  constexpr Floating high = std::numeric_limits<Floating>::max() / 2;
  return impl::fp_abs(a) <= high && impl::fp_abs(b) <= high
           ? // typical case: overflow is impossible
           (a + b) / 2
           :                                     // always correctly rounded
           impl::fp_abs(a) < low ? a + b / 2 :   // not safe to halve a
             impl::fp_abs(b) < low ? a / 2 + b : // not safe to halve b
               a / 2 + b / 2;                    // otherwise correctly rounded
}
#else
  using std::midpoint;
#endif


#if (!LIBPRESSIO_COMPAT_HAS_TRANSFORM_REDUCE)
  /**
   * transforms than reduces a sequence
   *
   * \param first iterator to the beginning of the container
   * \param last iterator to the end of the container
   * \param init initial value for the reduction
   * \param b Binary Operation to preform the reduction
   * \param u Unary Operation to transform the elements
   */
template <class InputIterator, class Type, class BinaryOp, class UnaryOp>
inline Type
transform_reduce(InputIterator first, InputIterator last, Type init, BinaryOp b,
                 UnaryOp u)
{
  for (; first != last; ++first)
    init = b(init, u(*first));
  return init;
}

  /**
   * transforms than reduces a sequence
   *
   * \param first1 iterator to the beginning of the container
   * \param last1 iterator to the end of the container
   * \param first2 iterator to the beginning of the container
   * \param init initial value for the reduction
   * \param b1 Binary Operation to preform the reduction
   * \param b2 Binary Operation to transform the elements together
   */
template <class InputIterator1, class InputIterator2, class Type,
          class BinaryOp1, class BinaryOp2>
inline Type
transform_reduce(InputIterator1 first1, InputIterator1 last1,
                 InputIterator2 first2, Type init, BinaryOp1 b1, BinaryOp2 b2)
{
  for (; first1 != last1; ++first1, (void)++first2)
    init = b1(init, b2(*first1, *first2));
  return init;
}


/**
 * computes a dot product
 *
 * \param first1 iterator to the beginning of the container
 * \param last1 iterator to the end of the container
 * \param first2 iterator to the beginning of the container
 * \param init initial value for the reduction
 */
template <class InputIterator1, class InputIterator2, class Type>
inline Type
transform_reduce(InputIterator1 first1, InputIterator1 last1,
                 InputIterator2 first2, Type init)
{
  return compat::transform_reduce(first1, last1, first2, std::move(init), compat::plus<>(),
                          compat::multiplies<>());
}
#else
using std::transform_reduce;
#endif

#if (!LIBPRESSIO_COMPAT_HAS_EXCLUSIVE_SCAN)
/**
 * computes an exclusive_scan
 *
 * \param first iterator to the beginning of the input
 * \param last iterator to the end of the input
 * \param d_first iterator to the beginning of the ouput
 * \param init initial value of the scan
 * \param binary_op the operation to scan over
 */
template <class InputIt, class OutputIt, class T, class BinaryOperation>
OutputIt
exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init,
               BinaryOperation binary_op)
{
  if (first != last) {
    T saved = init;
    do {
      init = binary_op(init, *first);
      *d_first = saved;
      saved = init;
      ++d_first;
    } while (++first != last);
  }

  return d_first;
}

/**
 * computes an exclusive prefix sum
 *
 * \param first iterator to the beginning of the input
 * \param last iterator to the end of the input
 * \param d_first iterator to the beginning of the ouput
 * \param init initial value of the scan
 */
template <class InputIt, class OutputIt, class T>
OutputIt
exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init)
{
  ::compat::exclusive_scan(first, last, d_first, init, plus<>{});
}
#else
using std::exclusive_scan;
#endif /*end exclusive_scan*/
}


#endif /* end of include guard: LIBPRESSIO_COMPAT_NUMERIC_H
 */
