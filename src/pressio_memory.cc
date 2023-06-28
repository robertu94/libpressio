#include "libpressio_ext/cpp/memory.h"
#include "libpressio_ext/domain/malloc.h"
#include "libpressio_ext/domain/nonowning.h"
#include <std_compat/utility.h>

pressio_memory &pressio_memory::operator=(pressio_memory const &rhs) {
  if (&rhs == this)
    return *this;

  if (data_capacity < rhs.data_capacity) {
    data_domain->free(data_ptr, data_capacity);
    data_ptr = data_domain->alloc(rhs.data_capacity);
    data_capacity = rhs.data_capacity;
  }
  size_t new_capacity = std::min(rhs.data_capacity, data_capacity);
  if (new_capacity) {
    // if accessible
    data_domain->memcpy(data_ptr, rhs.data_ptr, new_capacity);
    // else use copy-engine
  }

  return *this;
}
pressio_memory &pressio_memory::operator=(pressio_memory &&rhs) noexcept {
  if (&rhs == this)
    return *this;
  if (*rhs.domain() == *domain()) {
    // if we are in the same domains are equal, just move pointers
    data_ptr = compat::exchange(rhs.data_ptr, nullptr);
    data_capacity = compat::exchange(rhs.data_capacity, 0);
  } else {
    operator=(rhs);
  }

  return *this;
}

// default
pressio_memory::pressio_memory()
    : data_domain(std::make_shared<pressio_malloc_domain>()), data_ptr(nullptr), data_capacity(0) {}
pressio_memory::pressio_memory(std::shared_ptr<pressio_domain> &&domain)
    : data_domain(domain), data_ptr(nullptr), data_capacity(0) {}

// owning constructor
pressio_memory::pressio_memory(size_t n)
    : data_domain(std::make_shared<pressio_malloc_domain>()), data_ptr(data_domain->alloc(n)),
      data_capacity(n) {}

pressio_memory::pressio_memory(size_t n, std::shared_ptr<pressio_domain> &&domain)
    : data_domain(domain), data_ptr(data_domain->alloc(n)), data_capacity(n) {}

// non-owning constructor
pressio_memory::pressio_memory(void *ptr, size_t n)
    : data_domain(std::make_shared<pressio_nonowning_domain>()), data_ptr(ptr), data_capacity(n) {}

pressio_memory::pressio_memory(void *ptr, size_t n, std::shared_ptr<pressio_domain> &&domain)
    : data_domain(domain), data_ptr(ptr), data_capacity(n) {}

// copy constructor
pressio_memory::pressio_memory(pressio_memory const &rhs)
    : data_domain(std::make_shared<pressio_malloc_domain>()),
      data_ptr(data_domain->alloc(rhs.capacity())), data_capacity(rhs.capacity()) {
  operator=(rhs);
}

pressio_memory::pressio_memory(pressio_memory const &rhs, std::shared_ptr<pressio_domain> &&domain)
    : data_domain(domain), data_ptr(data_domain->alloc(rhs.capacity())),
      data_capacity(rhs.capacity()) {
  operator=(rhs);
}
// move constructor
pressio_memory::pressio_memory(pressio_memory &&rhs) noexcept
    : data_domain(rhs.data_domain->clone()), data_ptr(compat::exchange(rhs.data_ptr, nullptr)),
      data_capacity(compat::exchange(rhs.data_capacity, 0)) {}
pressio_memory::pressio_memory(pressio_memory &&rhs, std::shared_ptr<pressio_domain> &&domain)
    : data_domain(std::move(domain)), data_ptr(domain->alloc(rhs.capacity())),
      data_capacity(rhs.capacity()) {
  operator=(rhs);
}

void *pressio_memory::data() const { return data_ptr; }
size_t pressio_memory::capacity() const { return data_capacity; }
std::shared_ptr<pressio_domain> pressio_memory::domain() const { return data_domain; }
