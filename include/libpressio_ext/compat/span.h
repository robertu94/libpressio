/**
 * \file
 * \brief back ports of `<span>`
 */
//
//
// functions in this file are adapted from libc++-v10.0.1 whose license is
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

#ifndef LIBPRESSIO_COMPAT_SPAN_H
#define LIBPRESSIO_COMPAT_SPAN_H
#include <pressio_version.h>
#include <cassert>
#include <limits>
#include <array>
#include <tuple>
#include <type_traits>
#include "type_traits.h"
#include "iterator.h"
#include "cstddef.h"

#if LIBPRESSIO_COMPAT_HAS_SPAN
#include <span>
#endif


namespace compat {
#if !(LIBPRESSIO_COMPAT_HAS_SPAN)
/// tag indicating span has a dynamic_extent
extern const size_t dynamic_extent;
/// macro indicating span has a dynamic_extent; use when a inline compile time constant is required
#define compat_dynamic_extent std::numeric_limits<size_t>::max()

///forward declare template for span
template <typename Type, size_t Extent = compat_dynamic_extent> class span;


namespace {
template <class Type>
struct is_span_impl : public std::false_type {};

template <class Type, size_t Extent>
struct is_span_impl<span<Type, Extent>> : public std::true_type {};

template <class Type>
struct is_span : public is_span_impl<typename std::remove_cv<Type>::type> {};

template <class Type>
struct is_std_array_impl : public std::false_type {};

template <class Type, size_t Size>
struct is_std_array_impl<std::array<Type, Size>> : public std::true_type {};

template <class Type>
struct is_std_array : public is_std_array_impl<typename std::remove_cv<Type>::type> {};

template <class Type, class ElementType, class = void>
struct is_span_compatible_container : public std::false_type {};

template <class Type, class ElementType>
struct is_span_compatible_container<Type, ElementType,
        compat::void_t<
        // is not a specialization of span
            typename std::enable_if<!is_span<Type>::value, std::nullptr_t>::type,
        // is not a specialization of std::array
            typename std::enable_if<!is_std_array<Type>::value, std::nullptr_t>::type,
        // is_array_v<Container> is false,
            typename std::enable_if<!std::is_array<Type>::value, std::nullptr_t>::type,
        // data(cont) and size(cont) are well formed
            decltype(compat::data(std::declval<Type>())),
            decltype(compat::size(std::declval<Type>())),
        // remove_pointer_t<decltype(data(cont))>(*)[] is convertible to ElementType(*)[]
            typename std::enable_if<
                std::is_convertible<
                  decltype(compat::data(std::declval<Type &>()))
                  , typename std::add_pointer<ElementType>::type
                >::value,
                std::nullptr_t>::type
        >>
    : public std::true_type {};
}


/**
 * A non-owning span
 */
template <typename Type, size_t Extent>
class  span {
public:
//  constants and types
    using element_type           = Type;
    using value_type             = std::remove_cv_t<Type>;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;
    using pointer                = Type *;
    using const_pointer          = const Type *;
    using reference              = Type &;
    using const_reference        = const Type &;
    using iterator               =  pointer;
    using const_iterator         =  const_pointer;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static constexpr size_type extent = Extent;

// [span.cons], span constructors, copy, assignment, and destructor
// 1
     constexpr span() noexcept : span_data{nullptr}
    { static_assert(Extent == 0, "Can't default construct a statically sized span with size > 0"); }

// 9
    constexpr span           (const span&) noexcept = default;
    constexpr span& operator=(const span&) noexcept = default;

// 2
     constexpr span(pointer ptr, size_type count) : span_data{ptr}
        { (void)count; assert(Extent == count && "size mismatch in span's constructor (ptr, len)"); }
// 3
     constexpr span(pointer first, pointer last) : span_data{first} {
       (void)last;
       assert(Extent == std::distance(first, last) && "size mismatch in span's constructor (ptr, ptr)");
     }

// 4
     constexpr span(element_type (&arr)[Extent])          noexcept : span_data{arr} {}
// 5
     constexpr span(      std::array<value_type, Extent>& arr) noexcept : span_data{arr.data()} {}
// 6
     constexpr span(const std::array<value_type, Extent>& arr) noexcept : span_data{arr.data()} {}

     template <class OtherElementType>

     constexpr span(const span<OtherElementType, Extent> &other,
                    typename std::enable_if<
                        std::is_convertible<typename std::add_pointer<OtherElementType>::type, typename std::add_pointer<element_type>::type >::value,
                        std::nullptr_t>::type = nullptr)
         : span_data{other.data()} {}

     template <class OtherElementType>

     constexpr span(const span<OtherElementType, compat_dynamic_extent> &other,
                    typename std::enable_if<
                        std::is_convertible<typename std::add_pointer<OtherElementType>, typename std::add_pointer<element_type>::type >::value,
                        std::nullptr_t>::type = nullptr) noexcept
         : span_data{other.data()} {
       assert(Extent == other.size() && "size mismatch in span's constructor (other span)");
     }

//  ~span() noexcept = default;

    template <size_t Count>
    
    constexpr span<element_type, Count> first() const noexcept
    {
        static_assert(Count <= Extent, "Count out of range in span::first()");
        return {data(), Count};
    }

    template <size_t Count>
    
    constexpr span<element_type, Count> last() const noexcept
    {
        static_assert(Count <= Extent, "Count out of range in span::last()");
        return {data() + size() - Count, Count};
    }

    
    constexpr span<element_type, compat_dynamic_extent> first(size_type count) const noexcept
    {
        assert(count <= size() && "Count out of range in span::first(count)");
        return {data(), count};
    }

    
    constexpr span<element_type, compat_dynamic_extent> last(size_type count) const noexcept
    {
        assert(count <= size() && "Count out of range in span::last(count)");
        return {data() + size() - count, count};
    }

    template <size_t _Offset, size_t Count = compat_dynamic_extent>
    
    constexpr auto subspan() const noexcept
        -> span<element_type, Count != compat_dynamic_extent ? Count : Extent - _Offset>
    {
        static_assert(_Offset <= Extent, "Offset out of range in span::subspan()");
        return {data() + _Offset, Count == compat_dynamic_extent ? size() - _Offset : Count};
    }


    
    constexpr span<element_type, compat_dynamic_extent>
       subspan(size_type offset, size_type count = compat_dynamic_extent) const noexcept
    {
        assert(offset <= size() && "Offset out of range in span::subspan(offset, count)");
        assert(count  <= size() || count == compat_dynamic_extent && "Count out of range in span::subspan(offset, count)");
        if (count == compat_dynamic_extent)
            return {data() + offset, size() - offset};
        assert(offset <= size() - count && "count + offset out of range in span::subspan(offset, count)");
        return {data() + offset, count};
    }

     constexpr size_type size()       const noexcept { return Extent; }
     constexpr size_type size_bytes() const noexcept { return Extent * sizeof(element_type); }
     constexpr bool empty()           const noexcept { return Extent == 0; }

     constexpr reference operator[](size_type idx) const noexcept {
       assert(idx >= 0 && idx < size() && "span<T,N>[] index out of bounds");
       return span_data[idx];
     }

     constexpr reference front() const noexcept
    {
        static_assert(Extent > 0, "span<T,N>[].front() on empty span");
        return span_data[0];
    }

     constexpr reference back() const noexcept
    {
        static_assert(Extent > 0, "span<T,N>[].back() on empty span");
        return span_data[size()-1];
    }

     constexpr pointer data()                         const noexcept { return span_data; }

// [span.iter], span iterator support
     constexpr iterator                 begin() const noexcept { return iterator(data()); }
     constexpr iterator                   end() const noexcept { return iterator(data() + size()); }
     constexpr const_iterator          cbegin() const noexcept { return const_iterator(data()); }
     constexpr const_iterator            cend() const noexcept { return const_iterator(data() + size()); }
     constexpr reverse_iterator        rbegin() const noexcept { return reverse_iterator(end()); }
     constexpr reverse_iterator          rend() const noexcept { return reverse_iterator(begin()); }
     constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
     constexpr const_reverse_iterator   crend() const noexcept { return const_reverse_iterator(cbegin()); }

     constexpr void swap(span &other) noexcept
    {
        pointer p = span_data;
        span_data = other.span_data;
        other.span_data = p;
    }

     span<const byte, Extent * sizeof(element_type)> impl_as_bytes() const noexcept
    { return {reinterpret_cast<const byte *>(data()), size_bytes()}; }

     span<byte, Extent * sizeof(element_type)> impl_as_writable_bytes() const noexcept
    { return {reinterpret_cast<byte *>(data()), size_bytes()}; }

private:
    pointer    span_data;

};


/**
 * A non-owning span
 */
template <typename Type>
class  span<Type, compat_dynamic_extent> {
private:

public:
//  constants and types
    using element_type           = Type;
    using value_type             = std::remove_cv_t<Type>;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;
    using pointer                = Type *;
    using const_pointer          = const Type *;
    using reference              = Type &;
    using const_reference        = const Type &;
    using iterator               =  pointer;
    using const_iterator         =  const_pointer;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static constexpr size_type extent = compat_dynamic_extent;

// [span.cons], span constructors, copy, assignment, and destructor
     constexpr span() noexcept : span_data{nullptr}, span_size{0} {}

    constexpr span           (const span&) noexcept = default;
    constexpr span& operator=(const span&) noexcept = default;

     constexpr span(pointer ptr, size_type count) : span_data{ptr}, span_size{count} {}
     constexpr span(pointer first, pointer last) : span_data{first}, span_size{static_cast<size_t>(std::distance(first, last))} {}

    template <size_t Size>
    constexpr span(element_type (&arr)[Size])          noexcept : span_data{arr}, span_size{Size} {}

    template <size_t Size>
    constexpr span(std::array<value_type, Size>& arr)       noexcept : span_data{arr.data()}, span_size{Size} {}

    template <size_t Size>
    
    constexpr span(const std::array<value_type, Size>& arr) noexcept : span_data{arr.data()}, span_size{Size} {}

    template <class Container>
        constexpr span(      Container& c,
            typename std::enable_if<is_span_compatible_container<Container, Type>::value, std::nullptr_t>::type = nullptr)
        : span_data{compat::data(c)}, span_size{(size_type) compat::size(c)} {}

    template <class Container>
        constexpr span(const Container& c,
            typename std::enable_if<is_span_compatible_container<const Container, Type>::value, std::nullptr_t>::type = nullptr)
        : span_data{compat::data(c)}, span_size{(size_type) compat::size(c)} {}

    template <class OtherElementType, size_t OtherExtent>
    constexpr span(const span<OtherElementType, OtherExtent> &other,
                   typename std::enable_if<
                       std::is_convertible<typename std::add_pointer<OtherElementType>::type, typename std::add_pointer<element_type>::type>::value,
                       std::nullptr_t>::type = nullptr) noexcept
        : span_data{other.data()}, span_size{other.size()} {}

    //    ~span() noexcept = default;

    template <size_t Count>
    constexpr span<element_type, Count> first() const noexcept
    {
        assert(Count <= size() && "Count out of range in span::first()");
        return {data(), Count};
    }

    template <size_t Count>
    constexpr span<element_type, Count> last() const noexcept
    {
        assert(Count <= size() && "Count out of range in span::last()");
        return {data() + size() - Count, Count};
    }

    
    constexpr span<element_type, compat_dynamic_extent> first(size_type count) const noexcept
    {
        assert(count <= size() && "Count out of range in span::first(count)");
        return {data(), count};
    }

    
    constexpr span<element_type, compat_dynamic_extent> last (size_type count) const noexcept
    {
        assert(count <= size() && "Count out of range in span::last(count)");
        return {data() + size() - count, count};
    }

    template <size_t _Offset, size_t Count = compat_dynamic_extent>
    
    constexpr span<Type, compat_dynamic_extent> subspan() const noexcept
    {
        assert(_Offset <= size() && "Offset out of range in span::subspan()");
        assert(Count == compat_dynamic_extent || _Offset + Count <= size() && "Count out of range in span::subspan()");
        return {data() + _Offset, Count == compat_dynamic_extent ? size() - _Offset : Count};
    }

    constexpr span<element_type, compat_dynamic_extent>
    
    subspan(size_type offset, size_type count = compat_dynamic_extent) const noexcept
    {
        assert(offset <= size() && "Offset out of range in span::subspan(offset, count)");
        assert((count  <= size() || count == compat_dynamic_extent) && "count out of range in span::subspan(offset, count)");
        if (count == compat_dynamic_extent)
            return {data() + offset, size() - offset};
        assert(offset <= size() - count && "Offset + count out of range in span::subspan(offset, count)");
        return {data() + offset, count};
    }

     constexpr size_type size()       const noexcept { return span_size; }
     constexpr size_type size_bytes() const noexcept { return span_size * sizeof(element_type); }
     constexpr bool empty()           const noexcept { return span_size == 0; }

     constexpr reference operator[](size_type idx) const noexcept
    {
        assert(idx >= 0 && idx < size() && "span<T>[] index out of bounds");
        return span_data[idx];
    }

     constexpr reference front() const noexcept
    {
        assert(!empty() && "span<T>[].front() on empty span");
        return span_data[0];
    }

     constexpr reference back() const noexcept
    {
        assert(!empty() && "span<T>[].back() on empty span");
        return span_data[size()-1];
    }


     constexpr pointer data()                         const noexcept { return span_data; }

// [span.iter], span iterator support
     constexpr iterator                 begin() const noexcept { return iterator(data()); }
     constexpr iterator                   end() const noexcept { return iterator(data() + size()); }
     constexpr const_iterator          cbegin() const noexcept { return const_iterator(data()); }
     constexpr const_iterator            cend() const noexcept { return const_iterator(data() + size()); }
     constexpr reverse_iterator        rbegin() const noexcept { return reverse_iterator(end()); }
     constexpr reverse_iterator          rend() const noexcept { return reverse_iterator(begin()); }
     constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
     constexpr const_reverse_iterator   crend() const noexcept { return const_reverse_iterator(cbegin()); }

     constexpr void swap(span &other) noexcept
    {
        pointer p = span_data;
        span_data = other.span_data;
        other.span_data = p;

        size_type sz = span_size;
        span_size = other.span_size;
        other.span_size = sz;
    }

     span<const byte, compat_dynamic_extent> impl_as_bytes() const noexcept
    { return {reinterpret_cast<const byte *>(data()), size_bytes()}; }

     span<byte, compat_dynamic_extent> impl_as_writable_bytes() const noexcept
    { return {reinterpret_cast<byte *>(data()), size_bytes()}; }

private:
    pointer   span_data;
    size_type span_size;
};




//  as_bytes & as_writable_bytes
template <class Type, size_t Extent>

auto as_bytes(span<Type, Extent> s) noexcept
-> decltype(s.impl_as_bytes())
{ return    s.impl_as_bytes(); }

template <class Type, size_t Extent>

auto as_writable_bytes(span<Type, Extent> s) noexcept
-> typename std::enable_if<!std::is_const<Type>::value, decltype(s.impl_as_writable_bytes())>::type
{ return s.impl_as_writable_bytes(); }

template <class Type, size_t Extent>

constexpr void swap(span<Type, Extent> &lhs, span<Type, Extent> &rhs) noexcept
{ lhs.swap(rhs); }

#else
  using std::span;
#define compat_dynamic_extent std::dynamic_extent;
#endif
}

#endif /* end of include guard: LIBPRESSIO_COMPAT_SPAN_H */
