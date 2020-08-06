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
#ifndef PRESSIO_COMPAT
#define PRESSIO_COMPAT

#include "algorithm.h"
#include "cstddef.h"
#include "functional.h"
#include "iterator.h"
#include "memory.h"
#include "numeric.h"
#include "optional.h"
#include "span.h"
#include "type_traits.h"
#include "utility.h"
#include "variant.h"

#endif /*end header guard*/

///@endcond
