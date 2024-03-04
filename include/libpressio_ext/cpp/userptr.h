#ifndef PRESSIO_OPTIONS_USERPTR_CPP
#define PRESSIO_OPTIONS_USERPTR_CPP
#include <std_compat/utility.h>

/**
 * \file
 * \brief C++ inteface to user managed pointer objects (e.g. MPI_Comm, CudaStream_t, etc...)
 */

extern "C" {
void static_deleter(void*, void*);
void static_copy(void**, void**, const void*, const void*);
}

using userdata_deleter_t = void(*)(void*, void*);
template <class T>
userdata_deleter_t newdelete_deleter() {
  return [](void* ptr, void*) {
  T* ptr_typed = static_cast<T*>(ptr);
  delete ptr_typed;
  };
}

using userdata_copy_t = void(*)(void** dst, void**, const void* src, const void*);
template <class T>
userdata_copy_t newdelete_copy() {
  return [](void** dst, void**, const void* src, const void*) {
  T const* src_typed = static_cast<T const*>(src);
  T** dst_typed = reinterpret_cast<T**>(dst);
  *dst_typed = new T(*src_typed);
  };
}

/**
 * pointer type used to manage the lifetimes of user-defined pointer types in pressio_options
 */
class userdata {
  public:
    /**
     * default initalize with a nullptr
     */
    userdata():
      ptr(nullptr), metadata(nullptr), deleter(nullptr), copy(nullptr) {}

    /**
     * initailize from a non-owning deleter and noop-copy
     */
    userdata(void* ptr): ptr(ptr), metadata(nullptr), deleter(static_deleter), copy(static_copy) {}

    /**
     * initailize from pointers
     */
    userdata(void *ptr, void *metadata, void (*deleter)(void *, void *),
             void (*copy)(void **, void **, const void *, const void *))
        : ptr(ptr), metadata(metadata), deleter(deleter), copy(copy) {}
    /**
     * move initalizes the userdata, invoking the deleter
     * \param[in] data the data to be copied
     */
    userdata(userdata const &data): ptr(nullptr), metadata(nullptr), deleter(data.deleter), copy(data.copy) {
      if(data.copy) {
        data.copy(&ptr, &metadata, data.ptr, data.metadata); 
      }
    }
    /**
     * move initalizes the userdata, invoking the deleter
     * \param[in] data the data to be copied
     */
    userdata(userdata &&data) noexcept:
        ptr(compat::exchange(data.ptr,nullptr)),
        metadata(compat::exchange(data.metadata,nullptr)),
        deleter(compat::exchange(data.deleter, nullptr)),
        copy(compat::exchange(data.copy, nullptr)) {}
    /**
     * copy assigns the userdata, invoking the deleter
     * \param[in] data the data to be copied
     */
    userdata &operator=(userdata const &data) {
      if (&data == this) return *this;
      if(deleter) {
        deleter(ptr, metadata);
      }
      if(data.copy) {
        data.copy(&ptr, &metadata, &data.ptr, &data.metadata);
      }
      copy = data.copy;
      deleter = data.deleter;
      return *this;
    }
    /**
     * move assigns the userdata, taking ownership, and setting the donor's values to nullptr
     * \param[in] data the data to be copied
     */
    userdata& operator=(userdata&& data) noexcept {
      if (&data == this) return *this;
      if(deleter) {
          deleter(ptr, metadata);
      }
      ptr = compat::exchange(data.ptr, nullptr);
      metadata = compat::exchange(data.metadata, nullptr);
      copy = compat::exchange(data.copy, nullptr);
      deleter = compat::exchange(data.deleter, nullptr);
      return *this;
    }
    /**
     * \returns true the pointer is non-null
     */
    operator bool() const noexcept {
      return ptr;
    }
    /**
     * \returns true if pointers are equal
     */
    bool operator==(userdata const& data) const noexcept {
      return ptr == data.ptr;
    }

    ~userdata() {
      if(deleter != nullptr) {
        deleter(ptr, metadata);
      }
    }

  /**
   * return the raw pointer from the userdata
   */
  void* get() const {
    return ptr;
  }

  private:
  void* ptr;
  void* metadata;
  void(*deleter)(void*, void*);
  void(*copy)(void**, void**, const void*, const void*);
};
#endif
