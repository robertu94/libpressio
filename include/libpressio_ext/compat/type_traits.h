/**
 * \file
 * \brief back ports of `<type_traits>`
 */
#ifndef LIBPRESSIO_COMPAT_TYPE_TRAITS_H
#define LIBPRESSIO_COMPAT_TYPE_TRAITS_H
#include <pressio_version.h>
#include  <type_traits>
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

namespace compat {
#if !(LIBPRESSIO_COMPAT_HAS_VOID_T)
  /**
   * maps a list of types to void, useful for SFINAE
   */
  template<typename... Ts> struct make_void { typedef void type;};
  template<typename... Ts> using void_t = typename make_void<Ts...>::type;
#else
  using std::void_t;
#endif

#if !(LIBPRESSIO_COMPAT_HAS_NEGATION)
  /**
   * negates a compile time constant
   */
  template<class B>
  struct negation : std::integral_constant<bool, !bool(B::value)> {};
#else
  using std::negation;
#endif

#if !(LIBPRESSIO_COMPAT_HAS_CONJUNCTION)
  /**
   * takes the conjunction of several compile time constants
   */
  template<class...> struct conjunction : std::true_type { };
  template<class B1> struct conjunction<B1> : B1 { };
  template<class B1, class... Bn>
  struct conjunction<B1, Bn...> 
    : std::conditional<bool(B1::value), typename conjunction<Bn...>::type, B1>::type {};

  template<class...> struct disjunction : std::false_type { };
  template<class B1> struct disjunction<B1> : B1 { };
  template<class B1, class... Bn>
  struct disjunction<B1, Bn...> 
    : std::conditional<bool(B1::value), B1, disjunction<Bn...>>::type  { };
#else
  using std::conjunction;
  using std::disjunction;
#endif

#if !(LIBPRESSIO_COMPAT_HAS_IS_NULL_POINTER)
  template< class T >
  struct is_null_pointer : std::is_same<std::nullptr_t, typename std::remove_cv<T>::type> {};
#else
  using std::is_null_pointer;
#endif
}

#endif /* end of include guard: LIBPRESSIO_COMPAT_TYPE_TRAITS_H */
