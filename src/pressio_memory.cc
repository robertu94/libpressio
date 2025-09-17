#include "libpressio_ext/cpp/memory.h"
#include <std_compat/utility.h>

pressio_memory &pressio_memory::operator=(pressio_memory const &rhs) {
  if (&rhs == this)
    return *this;

  if (data_domain && data_capacity < rhs.data_capacity) {
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
  if (&rhs == this){
    return *this;
  }
  data_ptr = compat::exchange(rhs.data_ptr, nullptr);
  data_capacity = compat::exchange(rhs.data_capacity, 0);
  data_domain = compat::exchange(rhs.data_domain, nullptr);

  return *this;
}

pressio_memory::~pressio_memory() {
    if(data_ptr && data_domain) {
        data_domain->free(data_ptr, data_capacity);
    }
}

// default
pressio_memory::pressio_memory()
    : data_domain(libpressio::domain_plugins().build("malloc")), data_ptr(nullptr), data_capacity(0) {}
pressio_memory::pressio_memory(std::shared_ptr<libpressio::domains::pressio_domain> &&domain)
    : data_domain(domain), data_ptr(nullptr), data_capacity(0) {}
pressio_memory::pressio_memory(std::shared_ptr<libpressio::domains::pressio_domain> const&domain)
    : data_domain(domain), data_ptr(nullptr), data_capacity(0) {}

// owning constructor
pressio_memory::pressio_memory(size_t n)
    : data_domain(libpressio::domain_plugins().build("malloc")), data_ptr(data_domain->alloc(n)),
      data_capacity(n) {}

pressio_memory::pressio_memory(size_t n, std::shared_ptr<libpressio::domains::pressio_domain> &&domain)
    : data_domain(domain), data_ptr(this->data_domain->alloc(n)), data_capacity(n) {}
pressio_memory::pressio_memory(size_t n, std::shared_ptr<libpressio::domains::pressio_domain> const&domain)
    : data_domain(domain), data_ptr(this->data_domain->alloc(n)), data_capacity(n) {}

// non-owning constructor
pressio_memory::pressio_memory(void *ptr, size_t n)
    : data_domain(libpressio::domain_plugins().build("nonowning")), data_ptr(ptr), data_capacity(n) {}

pressio_memory::pressio_memory(void *ptr, size_t n, std::shared_ptr<libpressio::domains::pressio_domain> &&domain)
    : data_domain(domain), data_ptr(ptr), data_capacity(n) {}
pressio_memory::pressio_memory(void *ptr, size_t n, std::shared_ptr<libpressio::domains::pressio_domain> const&domain)
    : data_domain(domain), data_ptr(ptr), data_capacity(n) {}

// copy constructor
pressio_memory::pressio_memory(pressio_memory const &rhs)
    : data_domain(rhs.data_domain->clone()),
      data_ptr(this->data_domain->alloc(rhs.capacity())), data_capacity(rhs.capacity()) {
  operator=(rhs);
}

pressio_memory::pressio_memory(pressio_memory const &rhs, std::shared_ptr<libpressio::domains::pressio_domain> &&domain)
    : data_domain(domain), data_ptr(this->data_domain->alloc(rhs.capacity())),
      data_capacity(rhs.capacity()) {
  operator=(rhs);
}
pressio_memory::pressio_memory(pressio_memory const &rhs, std::shared_ptr<libpressio::domains::pressio_domain> const&domain)
    : data_domain(domain), data_ptr(this->data_domain->alloc(rhs.capacity())),
      data_capacity(rhs.capacity()) {
  operator=(rhs);
}
// move constructor
pressio_memory::pressio_memory(pressio_memory &&rhs) noexcept
    : data_domain(compat::exchange(rhs.data_domain, nullptr)), data_ptr(compat::exchange(rhs.data_ptr, nullptr)),
      data_capacity(compat::exchange(rhs.data_capacity, 0)) {}
pressio_memory::pressio_memory(pressio_memory &&rhs, std::shared_ptr<libpressio::domains::pressio_domain> &&domain)
    : data_domain(std::move(domain)), data_ptr(this->data_domain->alloc(rhs.capacity())),
      data_capacity(rhs.capacity()) {
  operator=(rhs);
}
pressio_memory::pressio_memory(pressio_memory &&rhs, std::shared_ptr<libpressio::domains::pressio_domain> const&domain)
    : data_domain(domain), data_ptr(this->data_domain->alloc(rhs.capacity())),
      data_capacity(rhs.capacity()) {
  operator=(rhs);
}

void *pressio_memory::data() const { return data_ptr; }
void *pressio_memory::release() { auto ptr = data_ptr; data_ptr = nullptr; return ptr; }
void pressio_memory::reset(void* ptr) { 
    if(data_ptr && data_domain) {
        data_domain->free(data_ptr, data_capacity);
    }
    data_ptr = ptr;
}
size_t pressio_memory::capacity() const { return data_capacity; }
std::shared_ptr<libpressio::domains::pressio_domain> const& pressio_memory::domain() const { return data_domain; }
