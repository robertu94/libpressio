#ifndef ROI_H_5ICRXVEF
#define ROI_H_5ICRXVEF 
#include <cstdint>
#include <array>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <iterator>

#include <libpressio_ext/cpp/data.h>

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

  basic_indexer(std::initializer_list<SizeType> args) noexcept: max_dims([](std::initializer_list<SizeType> args){
        std::array<SizeType,N> dims;
        std::copy(args.begin(), args.end(), dims.begin());
        return dims;
      }(args)) {}

  template <class It>
  basic_indexer(It first, It second) noexcept:
    max_dims([](It first, It second){
        std::array<SizeType,N> dims;
        std::copy(first, second, dims.begin());
        return dims;
      }(first, second)) {
    }

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


template <class T>
void copy_center(
    indexer<1> const &id,
    indexer<1> const &roi_size,
    indexer<2> const &roi,
    std::size_t const* center_ptr,
    std::size_t const center_idx,
    T const* origin,
    T * roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,1> center{center_ptr[0]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  //i,j,k
  for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
    const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
    if(i >= 0 && i < s_id[0]) {
      roi_mem[roi(mem_i, center_idx)] = origin[id(i)];
    } else {
      roi_mem[roi(mem_i, center_idx)] = 0;
    }
  }
}

template <class T>
void copy_center(
    indexer<2> const &id,
    indexer<2> const &roi_size,
    indexer<3> const &roi,
    std::size_t const* center_ptr,
    std::size_t const center_idx,
    T const* origin,
    T * roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,2> center{center_ptr[0], center_ptr[1]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  //i,j,k
  for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
    const ssize_t j = mem_j - s_roi_size[1] + s_center[1];
    if(j >= 0 && j < s_id[1]) {
      for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
        const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
        if(i >= 0 && i < s_id[0]) {
          roi_mem[roi(mem_i, mem_j, center_idx)] = origin[id(i,j)];
        } else {
          roi_mem[roi(mem_i, mem_j, center_idx)] = 0;
        }
      }
    } else {
      for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
        roi_mem[roi(mem_i, mem_j,center_idx)] = 0;
      }
    }
  }
}

template <class T>
void copy_center(
    indexer<3> const &id,
    indexer<3> const &roi_size,
    indexer<4> const &roi,
    size_t const* center_ptr,
    std::size_t const center_idx,
    T const* origin,
    T * roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,3> center{center_ptr[0], center_ptr[1], center_ptr[2]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  //i,j,k
  for (ssize_t mem_k = 0; mem_k < s_roi[2]; ++mem_k) {
    const ssize_t k = mem_k - s_roi_size[2] + s_center[2];
    if(k >= 0 && k < s_id[2]) {
      for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
        const ssize_t j = mem_j - s_roi_size[1] + s_center[1];
        if(j >= 0 && j < s_id[1]) {
          for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
            const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
            if(i >= 0 && i < s_id[0]) {
              roi_mem[roi(mem_i, mem_j, mem_k, center_idx)] = origin[id(i, j, k)];
            } else {
              roi_mem[roi(mem_i, mem_j, mem_k, center_idx)] = 0;
            }
          }
        } else {
          for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
            roi_mem[roi(mem_i, mem_j, mem_k, center_idx)] = 0;
          }
        }
      }
    } else {
      for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
        for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
          roi_mem[roi(mem_i, mem_j, mem_k, center_idx)] = 0;
        }
      }
    }
  }
}

template <class T>
void copy_center(
    indexer<4> const &id,
    indexer<4> const &roi_size,
    indexer<5> const &roi,
    size_t const* center_ptr,
    std::size_t const center_idx,
    T const* origin,
    T * roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,4> center{center_ptr[0], center_ptr[1], center_ptr[2], center_ptr[3]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  //i,j,k,l
  for (ssize_t mem_l = 0; mem_l < s_roi[3]; ++mem_l) {
    const ssize_t l = mem_l - s_roi_size[3] + s_center[3];
    if(l >= 0 && l < s_id[3]) {
      for (ssize_t mem_k = 0; mem_k < s_roi[2]; ++mem_k) {
        const ssize_t k = mem_k - s_roi_size[2] + s_center[2];
        if(k >= 0 && k < s_id[2]) {
          for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
            const ssize_t j = mem_j - s_roi_size[1] + s_center[1];
            if(j >= 0 && j < s_id[1]) {
              for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
                const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
                if(i >= 0 && i < s_id[0]) {
                  roi_mem[roi(mem_i, mem_j, mem_k, mem_l, center_idx)] = origin[id(i, j, k, l)];
                } else {
                  roi_mem[roi(mem_i, mem_j, mem_k, mem_l, center_idx)] = 0;
                }
              }
            } else {
              for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
                roi_mem[roi(mem_i, mem_j, mem_k, mem_l ,center_idx)] = 0;
              }
            }
          }
        } else {
          for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
            for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
              roi_mem[roi(mem_i, mem_j, mem_k, mem_l ,center_idx)] = 0;
            }
          }
        }
      }
    } else {
      for (ssize_t mem_k = 0; mem_k < s_roi[2]; ++mem_k) {
          for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
            for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
              roi_mem[roi(mem_i, mem_j, mem_k, mem_l ,center_idx)] = 0;
            }
          }
      }
    }
  }
}

template <class T, std::size_t N>
void roi_save(indexer<N> const &id,
              indexer<N> const &roi_size,
              indexer<N+1> const &roi,
              pressio_data const& centers_range,
              T const* origin,
              T * roi_mem,
              size_t n_threads) {
  auto centers_width = centers_range.get_dimension(0);
  auto centers_size = centers_range.get_dimension(1);

#pragma omp parallel for num_threads(n_threads)
  for (size_t i = 0; i < centers_size; ++i) {
    copy_center(id, roi_size, roi, static_cast<const size_t*>(centers_range.data()) + i*centers_width, i, origin, roi_mem);
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

template <class T>
void restore_center(
    indexer<1> const &id,
    indexer<1> const &roi_size,
    indexer<2> const &roi,
    size_t const* center_ptr,
    std::size_t const center_idx,
    T * origin,
    T const* roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,1> center{center_ptr[0]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
    const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
    if(i >= 0 && i < s_id[0]) {
      origin[id(i)] = roi_mem[roi(mem_i, center_idx)];
    } 
  }
}

template <class T>
void restore_center(
    indexer<2> const &id,
    indexer<2> const &roi_size,
    indexer<3> const &roi,
    size_t const* center_ptr,
    std::size_t const center_idx,
    T * origin,
    T const* roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,2> center{center_ptr[0], center_ptr[1]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
    const ssize_t j = mem_j - s_roi_size[1] + s_center[1];
    if(j >= 0 && j < s_id[1]) {
      for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
        const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
        if(i >= 0 && i < s_id[0]) {
          origin[id(i, j)] = roi_mem[roi(mem_i, mem_j, center_idx)];
        } 
      }
    } 
  }
}
template <class T>
void restore_center(
    indexer<3> const &id,
    indexer<3> const &roi_size,
    indexer<4> const &roi,
    std::size_t const* center_ptr,
    std::size_t const center_idx,
    T * origin,
    T const* roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,3> center{center_ptr[0], center_ptr[1], center_ptr[2]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  for (ssize_t mem_k = 0; mem_k < s_roi[2]; ++mem_k) {
    const ssize_t k = mem_k - s_roi_size[2] + s_center[2];
    if(k >= 0 && k < s_id[2]) {
      for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
        const ssize_t j = mem_j - s_roi_size[1] + s_center[1];
        if(j >= 0 && j < s_id[1]) {
          for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
            const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
            if(i >= 0 && i < s_id[0]) {
              origin[id(i, j, k)] = roi_mem[roi(mem_i, mem_j, mem_k, center_idx)];
            } 
          }
        } 
      }
    }
  }
}
template <class T>
void restore_center(
    indexer<4> const &id,
    indexer<4> const &roi_size,
    indexer<5> const &roi,
    std::size_t const* center_ptr,
    std::size_t const center_idx,
    T * origin,
    T const* roi_mem) {
  auto s_roi = as<ssize_t>(roi);
  auto s_roi_size = as<ssize_t>(roi_size);
  std::array<size_t,4> center{center_ptr[0], center_ptr[1], center_ptr[2], center_ptr[3]};
  auto s_center = as<ssize_t>(center);
  auto s_id = as<ssize_t>(id);

  for (ssize_t mem_l = 0; mem_l < s_roi[3]; ++mem_l) {
    const ssize_t l = mem_l - s_roi_size[3] + s_center[3];
    if(l >= 0 && l < s_id[3]) {
      for (ssize_t mem_k = 0; mem_k < s_roi[2]; ++mem_k) {
        const ssize_t k = mem_k - s_roi_size[2] + s_center[2];
        if(k >= 0 && k < s_id[2]) {
          for (ssize_t mem_j = 0; mem_j < s_roi[1]; ++mem_j) {
            const ssize_t j = mem_j - s_roi_size[1] + s_center[1];
            if(j >= 0 && j < s_id[1]) {
              for (ssize_t mem_i = 0; mem_i < s_roi[0]; ++mem_i) {
                const ssize_t i = mem_i - s_roi_size[0] + s_center[0];
                if(i >= 0 && i < s_id[0]) {
                  origin[id(i, j, k, l)] = roi_mem[roi(mem_i, mem_j, mem_k, mem_l, center_idx)];
                } 
              }
            } 
          }
        }
      }
    }
  }
}

template <class T, std::size_t N>
void roi_restore(indexer<N> const &id,
              indexer<N> const &roi_size,
              indexer<N+1> const &roi,
              pressio_data const& centers_range,
              T const* roi_mem,
              T * restored,
              size_t n_threads
              ) {

  auto centers_size = centers_range.get_dimension(1);
  auto centers_width = centers_range.get_dimension(0);

#pragma omp parallel for num_threads(n_threads)
  for (size_t i = 0; i < centers_size; ++i) {
    restore_center(id, roi_size, roi, static_cast<size_t*>(centers_range.data()) + centers_width*i, i, restored, roi_mem);
  }
}


template <class T>
auto bin_omp(indexer<1> const& id,  indexer<1> const& binned_storage, indexer<1> const& bins, T const* v, T* binned, size_t n_threads) {
#pragma omp parallel for  num_threads(n_threads)
    for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
      size_t i = i_s * bins[0];
      T sum = 0;
      size_t n = 0;
      for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
        if (b_i + i < id[0]) {
          ++n;
          sum += v[id(i + b_i)];
        }
      }
      binned[binned_storage(i_s)] = sum / n;
    }
}

template <class T>
auto bin_omp(indexer<2> const& id,  indexer<2> const& binned_storage, indexer<2> const& bins, T const* v, T* binned, size_t n_threads) {
#pragma omp parallel for collapse(2) num_threads(n_threads)
  for (size_t j_s = 0; j_s < binned_storage[1]; j_s++) {
    for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
      size_t j = j_s * bins[1];
      size_t i = i_s * bins[0];
      T sum = 0;
      size_t n = 0;
      for (size_t b_j = 0; b_j < bins[1]; ++b_j) {
        if (b_j + j < id[1]) {
          for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
            if (b_i + i < id[0]) {
              ++n;
              sum += v[id(i + b_i, j + b_j)];
            }
          }
        }
      }
      binned[binned_storage(i_s, j_s)] = sum / n;
    }
  }
}

template <class T>
auto bin_omp(indexer<3> const& id,  indexer<3> const& binned_storage, indexer<3> const& bins, T const* v, T* binned, size_t n_threads) {
#pragma omp parallel for collapse(3) num_threads(n_threads)
  for (size_t k_s = 0; k_s < binned_storage[2]; k_s++) {
    for (size_t j_s = 0; j_s < binned_storage[1]; j_s++) {
      for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
        size_t k = k_s * bins[2];
        size_t j = j_s * bins[1];
        size_t i = i_s * bins[0];
        T sum = 0;
        size_t n = 0;
        for (size_t b_k = 0; b_k < bins[2]; ++b_k) {
          if (b_k + k < id[2]) {
            for (size_t b_j = 0; b_j < bins[1]; ++b_j) {
              if (b_j + j < id[1]) {
                for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
                  if (b_i + i < id[0]) {
                    ++n;
                    sum += v[id(i + b_i, j + b_j, k + b_k)];
                  }
                }
              }
            }
          }
        }
        binned[binned_storage(i_s, j_s, k_s)] = sum / n;
      }
    }
  }
}

template <class T>
auto bin_omp(indexer<4> const& id,  indexer<4> const& binned_storage, indexer<4> const& bins, T const* v, T* binned, size_t n_threads) {
#pragma omp parallel for collapse(4) num_threads(n_threads)
  for (size_t l_s = 0; l_s < binned_storage[3]; l_s++) {
    for (size_t k_s = 0; k_s < binned_storage[2]; k_s++) {
      for (size_t j_s = 0; j_s < binned_storage[1]; j_s++) {
        for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
          size_t l = l_s * bins[3];
          size_t k = k_s * bins[2];
          size_t j = j_s * bins[1];
          size_t i = i_s * bins[0];
          T sum = 0;
          size_t n = 0;
          for (size_t b_l = 0; b_l < bins[3]; ++b_l) {
            if (b_l + l < id[3]) {
              for (size_t b_k = 0; b_k < bins[2]; ++b_k) {
                if (b_k + k < id[2]) {
                  for (size_t b_j = 0; b_j < bins[1]; ++b_j) {
                    if (b_j + j < id[1]) {
                      for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
                        if (b_i + i < id[0]) {
                          ++n;
                          sum += v[id(i + b_i, j + b_j, k + b_k, l+b_l)];
                        }
                      }
                    }
                  }
                }
              }
              binned[binned_storage(i_s, j_s, k_s, l_s)] = sum / n;
            }
          }
        }
      }
    }
  }
}

template <class T>
void restore_omp(indexer<1> const& id,  indexer<1> const& binned_storage, indexer<1> const& bins, T const* binned, T* restored, size_t n_threads) {
#pragma omp parallel for num_threads(n_threads)
  for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
    size_t i = i_s * bins[0];
    auto value = binned[binned_storage(i_s)];
    for (size_t b_l = 0; b_l < bins[3]; ++b_l) {
      for (size_t b_k = 0; b_k < bins[2]; ++b_k) {
        for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
          if (b_i + i < id[0]) {
            restored[id(i + b_i)] = value;
          }
        }
      }
    }
  }
}

template <class T>
void restore_omp(indexer<2> const& id,  indexer<2> const& binned_storage, indexer<2> const& bins, T const* binned, T* restored, size_t n_threads) {
#pragma omp parallel for collapse(2) num_threads(n_threads)
  for (size_t j_s = 0; j_s < binned_storage[1]; j_s++) {
    for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
      size_t j = j_s * bins[1];
      size_t i = i_s * bins[0];
      auto value = binned[binned_storage(i_s, j_s)];
      for (size_t b_l = 0; b_l < bins[3]; ++b_l) {
        for (size_t b_k = 0; b_k < bins[2]; ++b_k) {
          for (size_t b_j = 0; b_j < bins[1]; ++b_j) {
            if (b_j + j < id[1]) {
              for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
                if (b_i + i < id[0]) {
                  restored[id(i + b_i, j + b_j)] = value;
                }
              }
            }
          }
        }
      }
    }
  }
}

template <class T>
void restore_omp(indexer<3> const& id,  indexer<3> const& binned_storage, indexer<3> const& bins, T const* binned, T* restored, size_t n_threads) {
#pragma omp parallel for collapse(3) num_threads(n_threads)
  for (size_t k_s = 0; k_s < binned_storage[2]; k_s++) {
    for (size_t j_s = 0; j_s < binned_storage[1]; j_s++) {
      for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
        size_t k = k_s * bins[2];
        size_t j = j_s * bins[1];
        size_t i = i_s * bins[0];
        auto value = binned[binned_storage(i_s, j_s, k_s)];
        for (size_t b_l = 0; b_l < bins[3]; ++b_l) {
          for (size_t b_k = 0; b_k < bins[2]; ++b_k) {
            if (b_k + k < id[2]) {
              for (size_t b_j = 0; b_j < bins[1]; ++b_j) {
                if (b_j + j < id[1]) {
                  for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
                    if (b_i + i < id[0]) {
                      restored[id(i + b_i, j + b_j, k + b_k)] = value;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <class T>
void restore_omp(indexer<4> const& id,  indexer<4> const& binned_storage, indexer<4> const& bins, T const* binned, T* restored, size_t n_threads) {
#pragma omp parallel for collapse(4) num_threads(n_threads)
  for (size_t l_s = 0; l_s < binned_storage[3]; l_s++) {
    for (size_t k_s = 0; k_s < binned_storage[2]; k_s++) {
      for (size_t j_s = 0; j_s < binned_storage[1]; j_s++) {
        for (size_t i_s = 0; i_s < binned_storage[0]; i_s++) {
          size_t l = l_s * bins[3];
          size_t k = k_s * bins[2];
          size_t j = j_s * bins[1];
          size_t i = i_s * bins[0];
          auto value = binned[binned_storage(i_s, j_s, k_s, l_s)];
          for (size_t b_l = 0; b_l < bins[3]; ++b_l) {
            if (b_l + l < id[3]) {
              for (size_t b_k = 0; b_k < bins[2]; ++b_k) {
                if (b_k + k < id[2]) {
                  for (size_t b_j = 0; b_j < bins[1]; ++b_j) {
                    if (b_j + j < id[1]) {
                      for (size_t b_i = 0; b_i < bins[0]; ++b_i) {
                        if (b_i + i < id[0]) {
                          restored[id(i + b_i, j + b_j, k + b_k, l+b_l)] = value;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
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
