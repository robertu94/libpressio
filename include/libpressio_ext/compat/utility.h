/**
 * \file
 * \brief back ports of `<utility>`
 */
#ifndef LIBPRESSIO_COMPAT_UTILITY_H
#define LIBPRESSIO_COMPAT_UTILITY_H
#include <pressio_version.h>
#include <utility>

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
#if (!LIBPRESSIO_COMPAT_HAS_EXCHANGE)
  /**
   * exchanges two values, 
   * \param obj the value to modify
   * \param new_value the value to put in obj
   * \returns the old value in obj
   */
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
} // namespace compat

#endif /* end of include guard: LIBPRESSIO_COMPAT_UTILITY_H */
