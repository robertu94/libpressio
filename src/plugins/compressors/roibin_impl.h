#ifndef ROI_H_5ICRXVEF
#define ROI_H_5ICRXVEF 
#include <cstdint>
#include <array>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <iterator>

#include <std_compat/type_traits.h>
#include <std_compat/iterator.h>
#include <std_compat/functional.h>
#include <std_compat/span.h>

namespace libpressio { namespace roibin_ns {
//assume dims are fastest to slowest
template <class SizeType, SizeType N>
struct basic_indexer {
  template <class... T, typename std::enable_if<compat::conjunction<std::is_integral<typename std::decay<T>::type>...>::value,int>::type = 0>
  basic_indexer(T&&... args) noexcept: max_dims{static_cast<SizeType>(args)...} {}

  basic_indexer(std::array<SizeType,N> args) noexcept: max_dims(args) {}

  template <class It>
  basic_indexer(It first, It second) noexcept:
    max_dims([](It first, It second){
        std::array<SizeType,N> dims;
        std::copy(first, second, dims.begin());
        return dims;
      }(first, second)) {}

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




template <std::size_t N, class T>
void copy_center(
    indexer<N> const &id,
    indexer<N> const &roi_size,
    indexer<N+1> const &roi,
    std::array<std::size_t, N> const &center,
    std::size_t const center_idx,
    T const* origin,
    T * roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);


  for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
    for (ssize_t mem_i = 0; mem_i < s_roi[1]; ++mem_i) {
      ssize_t i = mem_i - (s_roi_size[0]) + s_center[0];
      ssize_t j = mem_j - (s_roi_size[1]) + s_center[1];
      if((i >= 0 && i < s_id[0]) && (j >= 0 && j < s_id[1])) {
        roi_mem[roi(mem_i,mem_j,0,0,center_idx)] = origin[id(i,j, center[2], center[3])];
      } else {
        roi_mem[roi(mem_i,mem_j,0,0,center_idx)] = 0;
      }
    }
  }
}

template <class T, std::size_t N, class CentersRange>
void roi_save(indexer<N> const &id,
              indexer<N> const &roi_size,
              indexer<N+1> const &roi,
              CentersRange const& centers_range,
              T const* origin,
              T * roi_mem) {
  auto centers_begin = std::begin(centers_range);
  auto centers_size = compat::size(centers_range);

#pragma omp parallel for
  for (size_t i = 0; i < centers_size; ++i) {
    copy_center(id, roi_size, roi, *std::next(centers_begin, i), i, origin, roi_mem);
  }
}

template <std::size_t N>
indexer<N+1> to_roimem(indexer<N> const& roi_size, std::size_t centers) {
 std::array<std::size_t, N+1> arr;
 for (size_t i = 0; i < N; ++i) {
    arr[i] = roi_size[i] * 2 + 1;
  }
  arr[N] = centers;
  return indexer<N+1>{arr};

}

template <std::size_t N, class T>
void restore_center(
    indexer<N> const &id,
    indexer<N> const &roi_size,
    indexer<N+1> const &roi,
    std::array<std::size_t, N> const &center,
    std::size_t const center_idx,
    T * origin,
    T const* roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);


  for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
    for (ssize_t mem_i = 0; mem_i < s_roi[1]; ++mem_i) {
      ssize_t i = mem_i - (s_roi_size[0]) + s_center[0];
      ssize_t j = mem_j - (s_roi_size[1]) + s_center[1];
      if((i >= 0 && i < s_id[0]) && (j >= 0 && j < s_id[1])) {
        origin[id(i,j, center[2], center[3])] = roi_mem[roi(mem_i,mem_j,0,0,center_idx)];
      }
    }
  }
}

template <class T, std::size_t N, class CentersRange>
void roi_restore(indexer<N> const &id,
              indexer<N> const &roi_size,
              indexer<N+1> const &roi,
              CentersRange const& centers_range,
              T const* roi_mem,
              T * restored
              ) {

  auto centers_begin = std::begin(centers_range);
  auto centers_size = compat::size(centers_range);

#pragma omp parallel for
  for (size_t i = 0; i < centers_size; ++i) {
    restore_center(id, roi_size, roi, *std::next(centers_begin, i), i, restored, roi_mem);
  }
}

template <class T, std::size_t N>
auto bin_omp(indexer<N> const& id,  indexer<N> const& binned_storage, indexer<N> const& bins, T const* v, T* binned) {
  #pragma omp parallel for collapse(2)
    for (std::size_t l = 0; l < id[3]; ++l) {
    for (std::size_t k = 0; k < id[2]; ++k) {
      for (std::size_t r = 0; r < binned_storage[1] ; ++r) {
      for (std::size_t c = 0; c < binned_storage[0]; ++c) {
        double sum = 0;
        unsigned n = 0;
        if(bins[0]*c*2 < id[0] && bins[1]*r*2 < id[1]) {
          //in bounds the whole time
          n = bins[0] *  bins[1];
          for (std::size_t ri = 0; ri < bins[1]; ++ri) {
          for (std::size_t ci = 0; ci < bins[0]; ++ci) {
              auto idx = id(c*bins[0]+ci, r*bins[1]+ri, k, l);
              sum += v[idx];
          }}
        } else {
          for (std::size_t ri = 0; ri < bins[1]; ++ri) {
          for (std::size_t ci = 0; ci < bins[0]; ++ci) {
            if(c*bins[0]+ci < id[0] && r*bins[1]+ri < id[1]) {
              auto idx = id(c*bins[0]+ci, r*bins[1]+ri, k, l);
              sum += v[idx];
              n++;
            }
          }}
        }
        auto binned_idx = binned_storage(c,r,k,l);
        binned[binned_idx] = sum/n;
      }}
    }}
}

template <class T, std::size_t N>
auto restore_omp(indexer<N> const& id,  indexer<N> const& binned_storage, indexer<N> const& bins, T const* binned, T* restored) {
  #pragma omp parallel for collapse(2)
    for (std::size_t l = 0; l < id[3]; ++l) {
    for (std::size_t k = 0; k < id[2]; ++k) {
      for (std::size_t r = 0; r < binned_storage[1] ; ++r) {
      for (std::size_t c = 0; c < binned_storage[0]; ++c) {
        auto binned_value = binned[binned_storage(c,r,k,l)];
        if(bins[0]*c*2 < id[0] && bins[1]*r*2 < id[1]) {
          //in bounds the whole time
          for (std::size_t ri = 0; ri < bins[1]; ++ri) {
          for (std::size_t ci = 0; ci < bins[0]; ++ci) {
              restored[id(c*bins[0]+ci, r*bins[1]+ri, k, l)] = binned_value;
          }}
        } else {
          for (std::size_t ri = 0; ri < bins[1]; ++ri) {
          for (std::size_t ci = 0; ci < bins[0]; ++ci) {
            if(c*bins[0]+ci < id[0] && r*bins[1]+ri < id[1]) {
              restored[id(c*bins[0]+ci, r*bins[1]+ri, k, l)] = binned_value;
            }
          }}
        }
      }}
    }}
}

template <size_t N>
indexer<N> to_binned_index(indexer<N> const& dims, indexer<N> const& bins) {
  std::array<std::size_t, N> binned_storage{};

  for (std::size_t i = 0; i < N; ++i) {
    std::size_t quot = dims[i] / bins[i];
    std::size_t rem = dims[i] % bins[i];
    binned_storage[i] = quot + ((rem == 0) ? 0:1);
  }

  return indexer<N>{binned_storage};
}

}}

#endif 
