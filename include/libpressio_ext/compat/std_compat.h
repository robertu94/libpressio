/**
 * \file 
 * \brief internal portability header
 * \details this header contains a couple of C++ standard algorithm replacements if
 * the provided standard library doesn't have them.  We prefer the standard
 * library versions if they exist.  Use of any of these functions outside of
 * libpressio may is NOT ALLOWED
 */

///@cond INTERNAL

//
// functions in this file are adapted from libc++-v9.0.0 whose license is
// reproduced below
//
// Copyright (c) 2009-2014 by the contributors listed in CREDITS.TXT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
//
#include <functional>
#include <numeric>
#include <memory>
#include <type_traits>
#include <pressio_version.h>

#ifndef PRESSIO_COMPAT
#define PRESSIO_COMPAT

#if !(LIBPRESSIO_COMPAT_HAS_OPTIONAL)
#include <boost/optional.hpp>
#else
#include <optional>
#endif

#if !(LIBPRESSIO_COMPAT_HAS_VARIANT)
#include <boost/variant.hpp>
#else
#include <variant>
#endif


namespace compat {

#if !(LIBPRESSIO_COMPAT_HAS_MULITPLIES)
template <class T = void>
struct multiplies
{
  template <class U, class V>
  constexpr auto operator()(U&& u, V&& v) const
    noexcept(noexcept(std::forward<U>(u) * std::forward<V>(v)))
      -> decltype(std::forward<U>(u) * std::forward<V>(v))
  {
    return std::forward<U>(u) * std::forward<V>(v);
  }
  using is_transparent = void;
};
template <class T = void>
struct plus
{
  template <class U, class V>
  constexpr auto operator()(U&& u, V&& v) const
    noexcept(noexcept(std::forward<U>(u) + std::forward<V>(v)))
      -> decltype(std::forward<U>(u) + std::forward<V>(v))
  {
    return u + v;
  }
  using is_transparent = void;
};
#else
using std::multiplies;
using std::plus;
#endif

#if (!LIBPRESSIO_COMPAT_HAS_EXCLUSIVE_SCAN)
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

template <class InputIt, class OutputIt, class T>
OutputIt
exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init)
{
  ::compat::exclusive_scan(first, last, d_first, init, plus<>{});
}
#else
using std::exclusive_scan;
#endif /*end exclusive_scan*/

#if (!LIBPRESSIO_COMPAT_HAS_TRANSFORM_REDUCE)
template <class InputIterator, class Type, class BinaryOp, class UnaryOp>
inline Type
transform_reduce(InputIterator first, InputIterator last, Type init, BinaryOp b,
                 UnaryOp u)
{
  for (; first != last; ++first)
    init = b(init, u(*first));
  return init;
}

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

#if (!LIBPRESSIO_COMPAT_HAS_EXCHANGE)
template<class T, class U = T>
  T exchange(T& obj, U&& new_value)
  {
      T old_value = std::move(obj);
      obj = std::forward<U>(new_value);
      return old_value;
  }
#else
  using std::exchange;
#endif

#if (!LIBPRESSIO_COMPAT_HAS_RBEGINEND)
  template< class C >
  auto rbegin( C& c ) -> decltype(c.rbegin()) {
    return c.rbegin();
  }
  template< class C >
  auto rend( C& c ) -> decltype(c.rend()) {
    return c.rend();
  }
  template< class C >
  auto rbegin( C const& c ) -> decltype(c.rbegin()) {
    return c.rbegin();
  }
  template< class C >
  auto rend( C const& c ) -> decltype(c.rend()) {
    return c.rend();
  }
#else
  using std::rbegin;
  using std::rend;
#endif

#if (!LIBPRESSIO_COMPAT_HAS_OPTIONAL)
  using boost::optional;
#else
#include<optional>
  using std::optional;
  using std::nullopt;
#endif


#if (!LIBPRESSIO_COMPAT_HAS_VARIANT)
  using boost::variant;
  using boost::get;
  struct monostate {};
  template <typename T, typename... Ts>
  bool holds_alternative(const boost::variant<Ts...>& v) noexcept
  {
      return boost::get<T>(&v) != nullptr;
  }
#else
  using std::variant;
  using std::monostate;
  using std::holds_alternative;
  using std::get;
#endif

#if (!LIBPRESSIO_COMPAT_HAS_MAKE_UNIQUE)
  template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args)
  {
      return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }
#else
  using std::make_unique;
#endif

#if !(LIBPRESSIO_COMPAT_HAS_CONJUNCTION)
  template<class...> struct conjunction : std::true_type { };
  template<class B1> struct conjunction<B1> : B1 { };
  template<class B1, class... Bn>
  struct conjunction<B1, Bn...> 
    : std::conditional<bool(B1::value), typename conjunction<Bn...>::type, B1>::type {};
#else
  using std::conjunction;
#endif

#if !(LIBPRESSIO_COMPAT_HAS_MIDPOINT)

template <class Type>
constexpr typename std::enable_if<std::is_integral<Type>::value &&
                                    !std::is_same<bool, Type>::value &&
                                    !std::is_null_pointer<Type>::value,
                                  Type>::type
midpoint(Type a, Type b) noexcept
{
  using Up = std::make_unsigned_t<Type>;
  constexpr Up bitshift = std::numeric_limits<Up>::digits - 1;

  Up diff = Up(b) - Up(a);
  Up sign_bit = b < a;

  Up half_diff = (diff / 2) + (sign_bit << bitshift) + (sign_bit & diff);

  return a + half_diff;
}

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
  template <typename Floating> constexpr Floating fp_abs(Floating f) { return f >= 0 ? f : -f; }
} // namespace impl

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

} // namespace compat

#endif /*end header guard*/

///@endcond
