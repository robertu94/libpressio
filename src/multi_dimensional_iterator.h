#ifndef PRESSIO_MULTI_DIMENSIONAL_ITERATOR
#define PRESSIO_MULTI_DIMENSIONAL_ITERATOR

#include <cstddef>
#include <memory>
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include "libpressio_ext/compat/std_compat.h"

namespace {
  template <class Type> 
  class cycle {
    public:
    cycle()=default;
    ~cycle()=default;
    cycle(cycle const&)=default;
    cycle& operator=(cycle const&)=default;
    cycle& operator=(cycle &&) noexcept =default;

    cycle(Type v): value(v) {}

    using value_type = Type;
    using difference_type = std::ptrdiff_t;
    using reference= Type const&;
    using pointer = Type*;
    using iterator_category = std::forward_iterator_tag;

    Type operator*(){ return value; }
    Type* operator->(){ return &value; }
    cycle& operator++(int) { return *this; }
    cycle operator++() { return *this; }
    bool operator==(cycle const& rhs){ return value==rhs.value; }
    bool operator!=(cycle const& rhs){ return value!=rhs.value; }

    private:
    Type value;
  };

}

template <class Type>
class multi_dimensional_range: public std::enable_shared_from_this<multi_dimensional_range<Type>> {
  public:

  class multi_dimensional_iterator {
    public:
      using value_type = Type;
      using difference_type = std::ptrdiff_t;
      using reference = Type&;
      using const_reference = Type const&;
      using pointer = Type*;
      using iterator_category = std::random_access_iterator_tag;

      ~multi_dimensional_iterator()=default;
      multi_dimensional_iterator()=default;
      multi_dimensional_iterator(multi_dimensional_iterator const&)=default;
      multi_dimensional_iterator& operator=(multi_dimensional_iterator const&)=default;
      multi_dimensional_iterator(multi_dimensional_iterator &&) noexcept=default;
      multi_dimensional_iterator& operator=(multi_dimensional_iterator &&) noexcept=default;
      multi_dimensional_iterator(std::shared_ptr<multi_dimensional_range>&& range, std::size_t current) noexcept:
        range(range), current(current) {}

      multi_dimensional_iterator& operator+=(std::ptrdiff_t rhs) {
        current+=rhs;
        return *this;
      }
      multi_dimensional_iterator& operator-=(std::ptrdiff_t n) { *this += -n; return *this;}
      multi_dimensional_iterator& operator--() { *this += -1 ; return *this;}
      multi_dimensional_iterator operator--(int) { auto cpy = *this; --(*this); return cpy; }
      multi_dimensional_iterator& operator++() { *this += 1; return *this;}
      multi_dimensional_iterator operator++(int) { auto cpy = *this; ++(*this); return cpy; }

      multi_dimensional_iterator operator+(std::ptrdiff_t n) const { auto cpy = *this; cpy += n; return cpy; }
      multi_dimensional_iterator operator-(std::ptrdiff_t n) const { auto cpy = *this; cpy -= n; return cpy; }
      difference_type operator-(multi_dimensional_iterator const& rhs) const {
        return static_cast<std::ptrdiff_t>(current) - static_cast<std::ptrdiff_t>(rhs.current);
      }

      reference operator[](std::ptrdiff_t) { return range->origin[range->current_to_offset(current)]; }
      const_reference operator[](std::ptrdiff_t) const { return range->origin[range->current_to_offset(current)]; }

      pointer operator->() {
        return range->origin + range->current_to_offset(current);
      }
      pointer operator->() const {
        return range->origin + range->current_to_offset(current);
      }
      reference operator*() {
        return range->origin[range->current_to_offset(current)];
      }
      const_reference operator*() const {
        return range->origin[range->current_to_offset(current)];
      }

      bool operator==(multi_dimensional_iterator const& rhs) const {
        return current == rhs.current;
      }
      bool operator!=(multi_dimensional_iterator const& rhs) const {
        return current != rhs.current;
      }
      bool operator<(multi_dimensional_iterator const& rhs) const {
        return current < rhs.current;
      }
      bool operator>(multi_dimensional_iterator const& rhs) const {
        return current > rhs.current;
      }
      bool operator>=(multi_dimensional_iterator const& rhs) const {
        return current >= rhs.current;
      }
      bool operator<=(multi_dimensional_iterator const& rhs) const {
        return current <= rhs.current;
      }

    private:
      friend multi_dimensional_range;
      std::shared_ptr<multi_dimensional_range> range;
      std::size_t current;
  };

  using iterator = multi_dimensional_iterator;
  using const_iterator = multi_dimensional_iterator;
  using value_type = Type;
  using reference = Type&;
  using pointer = Type*;

  multi_dimensional_iterator begin() {
    return multi_dimensional_iterator(this->shared_from_this(), 0);
  }
  multi_dimensional_iterator end() {
    return multi_dimensional_iterator(this->shared_from_this(), local_max_dim);
  }
  /**
   * constructs a multi_dimensional_range
   *
   * \param[in] global_dims the dimensions of the overall array
   * \param[in] stride how many items to skip in the global array in each direction
   * \param[in] count how many items to include from the global array in each direction
   */
  template <class ForwardIt1, class ForwardIt2 = cycle<size_t>, class ForwardIt3 = cycle<size_t>, class ForwardIt4 = cycle<size_t>>
  multi_dimensional_range(
      Type* origin,
      ForwardIt1 global_dims_begin,
      ForwardIt1 global_dims_end,
      ForwardIt2 count,
      ForwardIt3 stride = cycle<size_t>(1ul),
      ForwardIt4 start = cycle<size_t>(0ul)
      ): global_dims(global_dims_begin, global_dims_end),
         local_dims(count, std::next(count, global_dims.size())),
         local_stride(global_dims.size()),
         global_stride(global_dims.size()),
         local_max_dim(std::accumulate(count, std::next(count,global_dims.size()), 1, compat::multiplies<>{})),
         global_max_dim(std::accumulate(std::begin(global_dims), std::end(global_dims), 1, compat::multiplies<>{})),
         origin(origin)
  {
    static_assert(
        std::is_convertible<
          typename std::iterator_traits<ForwardIt1>::value_type,
          std::size_t>::value,
          "ForwardIt1 must be convertible to std::size_t"
    );
    static_assert(
        std::is_convertible<
          typename std::iterator_traits<ForwardIt2>::value_type,
          std::size_t>::value,
          "ForwardIt2 must be convertible to std::size_t"
    );
    static_assert(
        std::is_convertible<
          typename std::iterator_traits<ForwardIt3>::value_type,
          std::size_t>::value,
          "ForwardIt3 must be convertible to std::size_t"
    );
    static_assert(
        std::is_convertible<
          typename std::iterator_traits<ForwardIt4>::value_type,
          std::size_t>::value,
          "ForwardIt4 must be convertible to std::size_t"
    );

    //compute the global stride
    compat::exclusive_scan(std::begin(global_dims), std::end(global_dims), std::begin(global_stride),  1, compat::multiplies<>{});

    //compute the global offset without accounting for the stride
    global_offset = compat::transform_reduce(std::begin(global_stride), std::end(global_stride), start, 0);

    //change global stride to account for the stride
    std::transform(std::begin(global_stride), std::end(global_stride), stride, std::begin(global_stride), compat::multiplies<>{});

    compat::exclusive_scan(count, std::next(count,global_dims.size()), std::begin(local_stride), 1, compat::multiplies<>{});
    
  }

  template <class ForwardIt = cycle<size_t>, class ForwardIt2 = cycle<size_t>>
  multi_dimensional_range(
      multi_dimensional_iterator iterator,
      ForwardIt count,
      ForwardIt2 stride = cycle<size_t>(1ul)
      ): global_dims(iterator.range->global_dims),
         local_dims(count, std::next(count, global_dims.size())),
         local_stride(global_dims.size()),
         global_stride(global_dims.size()),
         local_max_dim(std::accumulate(count, std::next(count, global_dims.size()), 1, compat::multiplies<>{})),
         global_max_dim(iterator.range->global_max_dim),
         global_offset(iterator.range->current_to_offset(iterator.current)),
         origin(iterator.range->origin)
  {
    static_assert(
        std::is_convertible<
          typename std::iterator_traits<ForwardIt>::value_type,
          std::size_t>::value,
          "ForwardIt must be convertible to std::size_t"
    );
    static_assert(
        std::is_convertible<
          typename std::iterator_traits<ForwardIt2>::value_type,
          std::size_t>::value,
          "ForwardIt2 must be convertible to std::size_t"
    );
    compat::exclusive_scan(std::begin(global_dims), std::end(global_dims), std::begin(global_stride),  1, compat::multiplies<>{});
    std::transform(std::begin(global_stride), std::end(global_stride), stride, std::begin(global_stride), compat::multiplies<>{});

    compat::exclusive_scan(count, std::next(count, global_dims.size()), std::begin(local_stride), 1, compat::multiplies<>{});
    
  }

  template<class ForwardIt>
  typename std::enable_if<
    std::is_base_of<std::forward_iterator_tag,
      typename std::iterator_traits<ForwardIt>::iterator_category
    >::value,
    Type>::type
    operator()(ForwardIt requested_begin, ForwardIt requested_end)
  {
    if(std::distance(requested_begin, requested_end) != global_dims.size())
      throw std::runtime_error("invalid number of arguments passed to multi_dimensional_iterator");
    size_t current = compat::transform_reduce(
        std::begin(local_stride),
        std::end(local_stride),
        requested_begin,
        0
        );
    auto offset = current_to_offset(current);
    return origin[offset];
  }

  template<class... T>
  typename std::enable_if<
    compat::conjunction<std::is_convertible<T, size_t>...>::value
  ,Type>::type operator()(T... pos)
  {
    std::vector<size_t> requested = {static_cast<size_t>(pos)...};
    if(requested.size() != global_dims.size())
      throw std::runtime_error("invalid number of arguments passed to multi_dimensional_iterator");
    return this->operator()(requested.begin(), requested.end());
  }

  size_t num_dims() const { return global_dims.size(); };
  size_t get_global_dims(size_t idx) const { return global_dims.at(idx); };
  size_t get_local_dims(size_t idx) const { return local_dims.at(idx); };


  private:
    std::size_t current_to_offset(std::size_t current) {
      //first transform index to local_position
      std::vector<size_t> local_position(global_dims.size());
      std::transform(
          compat::rbegin(local_stride),
          compat::rend(local_stride),
          compat::rbegin(local_position),
          [&current](size_t stride){
            auto ret = current/stride;
            current -= (ret*stride);
            return ret;
          });
      //now compute global_offset
      return compat::transform_reduce(
          std::begin(global_stride),
          std::end(global_stride),
          std::begin(local_position),
          global_offset
          );
    }

    std::vector<std::size_t> global_dims; //the number of dimensions globally
    std::vector<std::size_t> local_dims; //the number of dimensions locally
    std::vector<std::size_t> local_stride; //if 1D, the number of elements between elements in the direction of each dimension in the view
    std::vector<std::size_t> global_stride; //if 1D, the number of elements between elements in the direction of each dimension in the global array
    std::size_t local_max_dim;
    std::size_t global_max_dim;
    std::size_t global_offset;
    Type* origin;
};


#endif
