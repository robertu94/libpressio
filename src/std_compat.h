/**
 * \file this header contains a couple of C++ standard algorithm replacements if
 * the provided standard library doesn't have them.  We prefer the standard
 * library versions if they exist
 */

//
// functions in this file are adapted from libc++-v8.0.0 whose license is
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

#ifndef PRESSIO_COMPAT
#define PRESSIO_COMPAT

namespace compat {

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
  ::compat::exclusive_scan(first, last, d_first, init, std::plus<>{});
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
  return compat::transform_reduce(first1, last1, first2, std::move(init), std::plus<>(),
                          std::multiplies<>());
}
#else
using std::transform_reduce;
#endif

} // namespace compat

#endif /*end header guard*/
