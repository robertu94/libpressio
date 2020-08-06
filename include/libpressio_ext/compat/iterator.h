/**
 * \file
 * \brief back ports of `<iterator>`
 */
#ifndef LIBPRESSIO_COMPAT_ITERATOR_H
#define LIBPRESSIO_COMPAT_ITERATOR_H
#include <pressio_version.h>
#include <iterator>
#include <initializer_list>

namespace compat {
#if (!LIBPRESSIO_COMPAT_HAS_RBEGINEND)
  /**
   * \param c container
   * \returns c.rbegin()
   */
  template< class C >
  auto rbegin( C& c ) -> decltype(c.rbegin()) {
    return c.rbegin();
  }
  /**
   * \param c container
   * \returns c.rend()
   */
  template< class C >
  auto rend( C& c ) -> decltype(c.rend()) {
    return c.rend();
  }
  /**
   * \param c container
   * \returns c.rbegin()
   */
  template< class C >
  auto rbegin( C const& c ) -> decltype(c.rbegin()) {
    return c.rbegin();
  }
  /**
   * \param c container
   * \returns c.rend()
   */
  template< class C >
  auto rend( C const& c ) -> decltype(c.rend()) {
    return c.rend();
  }
#else
  using std::rbegin;
  using std::rend;
#endif

#if (!LIBPRESSIO_COMPAT_HAS_DATA)
  /**
   * \param c container
   * \returns c.data()
   */
template <class C> 
constexpr auto data(C& c) -> decltype(c.data())
{
    return c.data();
}

/**
 * \param c container
 * \returns c.data()
 */
template <class C> 
constexpr auto data(const C& c) -> decltype(c.data())
{
    return c.data();
}

/**
 * \param array container
 * \returns array
 */
template <class T, std::size_t N>
constexpr T* data(T (&array)[N]) noexcept
{
    return array;
}

/**
 * \param il initializer_list
 * \returns il.begin
 */
template <class E> 
constexpr const E* data(std::initializer_list<E> il) noexcept
{
    return il.begin();
}
#else
  using std::data;
#endif

#if (!LIBPRESSIO_COMPAT_HAS_SIZE)
  /**
   * \param c container
   * \returns c.size()
   */
  template <class C> 
  constexpr auto size(const C& c) -> decltype(c.size())
  {
      return c.size();
  }

  /**
   * \param T a C array
   * \returns the size of the array
   */
  template <class T, std::size_t N>
  constexpr std::size_t size(const T (&)[N]) noexcept
  {
      return N;
  }
#else
  using std::size;
#endif
  
} /* compat */ 

#endif /* end of include guard: LIBPRESSIO_COMPAT_ITERATOR_H */
