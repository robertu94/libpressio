#include "python_memory_helpers.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include <cstdlib>
#include <cstring>
#include <new>

namespace {
int64_t allocations = 0;
int64_t bytes = 0;
int64_t sends = 0;

class python_tracking_domain final : public libpressio::domains::pressio_domain {
public:
  explicit python_tracking_domain(bool alternate): alternate(alternate) {}
  void* alloc(size_t n) override {
    void* ptr = std::malloc(n);
    if (!ptr) throw std::bad_alloc();
    ++allocations;
    bytes += n;
    // Initialize storage so Python can check sends without reading garbage.
    std::memset(ptr, 0, n);
    return ptr;
  }
  void free(void* ptr, size_t n) override {
    if (ptr) {
      --allocations;
      bytes -= n;
      std::free(ptr);
    }
  }
  void memcpy(void* dst, void* src, size_t n) override {
    std::memcpy(dst, src, n);
  }
  std::string const& prefix() const override {
    static const std::string primary = "python_test_tracking";
    static const std::string secondary = "python_test_alternate_tracking";
    return alternate ? secondary : primary;
  }
  std::shared_ptr<libpressio::domains::pressio_domain> clone() override {
    return std::make_shared<python_tracking_domain>(alternate);
  }
protected:
  bool equal(libpressio::domains::pressio_domain const& rhs) const noexcept override {
    auto other = dynamic_cast<python_tracking_domain const*>(&rhs);
    return other && alternate == other->alternate;
  }
private:
  bool alternate;
};

class python_tracking_send final : public libpressio::domains::pressio_domain_send {
public:
  void send(pressio_data& dst, pressio_data const& src) const override {
    ++sends;
    dst.domain()->memcpy(dst.data(), src.data(), src.size_in_bytes());
  }
};

libpressio::pressio_register primary_registration(libpressio::domain_plugins(),
    "python_test_tracking", [] { return std::make_shared<python_tracking_domain>(false); });
libpressio::pressio_register alternate_registration(libpressio::domain_plugins(),
    "python_test_alternate_tracking", [] { return std::make_shared<python_tracking_domain>(true); });
libpressio::pressio_register primary_send_registration(libpressio::domain_send_plugins(),
    "python_test_tracking>python_test_alternate_tracking",
    [] { return std::make_unique<python_tracking_send>(); });
libpressio::pressio_register alternate_send_registration(libpressio::domain_send_plugins(),
    "python_test_alternate_tracking>python_test_tracking",
    [] { return std::make_unique<python_tracking_send>(); });

std::shared_ptr<libpressio::domains::pressio_domain> domain(bool alternate) {
  return libpressio::domain_plugins().build(alternate
      ? "python_test_alternate_tracking" : "python_test_tracking");
}
}

int64_t memory_test_allocations() { return allocations; }
int64_t memory_test_bytes() { return bytes; }
int64_t memory_test_sends() { return sends; }
pressio_data* memory_test_new(bool alternate, bool empty) {
  auto result = std::make_unique<pressio_data>(empty
      ? pressio_data::empty(pressio_int32_dtype, {16}, domain(alternate))
      : pressio_data::owning(pressio_int32_dtype, {16}, domain(alternate)));
  if (!empty) {
    auto ptr = static_cast<int32_t*>(result->data());
    for (int32_t i = 0; i < 16; ++i) ptr[i] = i;
  }
  return result.release();
}
pressio_data* memory_test_readable_buffer(pressio_data* dst, pressio_data const* src) {
  return new pressio_data(domain_manager().make_readable(std::move(*dst), *src));
}
pressio_data* memory_test_readable_domain(bool alternate, bool move_domain,
                                        pressio_data const* src) {
  auto dst = domain(alternate);
  return new pressio_data(move_domain
      ? domain_manager().make_readable(std::move(dst), *src)
      : domain_manager().make_readable(dst, *src));
}
pressio_data* memory_test_writeable(bool alternate, bool move_source, pressio_data* src) {
  return new pressio_data(move_source
      ? domain_manager().make_writeable(domain(alternate), std::move(*src))
      : domain_manager().make_writeable(domain(alternate), *src));
}
void memory_test_move_assign(pressio_data* dst, pressio_data* src) {
  *dst = std::move(*src);
}
void memory_test_readable_alias_assign(pressio_data* data) {
  *data = domain_manager().make_readable(data->domain(), std::move(*data));
}
