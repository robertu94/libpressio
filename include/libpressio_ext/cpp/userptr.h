#ifndef PRESSIO_OPTIONS_USERPTR_CPP
#define PRESSIO_OPTIONS_USERPTR_CPP
#include <std_compat/utility.h>

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

class userdata {
  public:
    userdata():
      ptr(nullptr), metadata(nullptr), deleter(nullptr), copy(nullptr) {}

    userdata(void* ptr): ptr(ptr), metadata(nullptr), deleter(static_deleter), copy(static_copy) {}

    userdata(void *ptr, void *metadata, void (*deleter)(void *, void *),
             void (*copy)(void **, void **, const void *, const void *))
        : ptr(ptr), metadata(metadata), deleter(deleter), copy(copy) {}
    userdata(userdata const &data): ptr(nullptr), metadata(nullptr), deleter(data.deleter), copy(data.copy) {
      if(data.copy) {
        data.copy(&ptr, &metadata, data.ptr, data.metadata); 
      }
    }
    userdata(userdata &&data) noexcept:
        ptr(compat::exchange(data.ptr,nullptr)),
        metadata(compat::exchange(data.metadata,nullptr)),
        deleter(compat::exchange(data.deleter, nullptr)),
        copy(compat::exchange(data.copy, nullptr)) {}
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
    userdata& operator=(userdata&& data) noexcept {
      if (&data == this) return *this;
      ptr = compat::exchange(data.ptr, nullptr);
      metadata = compat::exchange(data.metadata, nullptr);
      copy = compat::exchange(data.copy, nullptr);
      deleter = compat::exchange(data.deleter, nullptr);
      return *this;
    }
    operator bool() const noexcept {
      return ptr;
    }
    bool operator==(userdata const& data) const noexcept {
      return ptr == data.ptr;
    }

    ~userdata() {
      if(deleter != nullptr) {
        deleter(ptr, metadata);
      }
    }

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
