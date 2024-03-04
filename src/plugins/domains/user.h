#ifndef PRESSIO_USER_DOMAIN_H
#define PRESSIO_USER_DOMAIN_H
#include "libpressio_ext/cpp/domain.h"
#include <functional>


/**
 * \file
 * \brief domain that uses user-defined C allocation and free functions
 */
struct pressio_user_domain: public pressio_domain, std::enable_shared_from_this<pressio_user_domain> {
    pressio_user_domain(void (*deleter)(void*, void*), void* metadata=nullptr):
        metadata_ptr(metadata)
    {
        if(deleter) {
            free_fn = [deleter](void* ptr, size_t, void* metadata){ deleter(ptr, metadata); };
        } else {
            free_fn = [](void*, size_t, void*){};
        }
    }
    pressio_user_domain(
            void* (*allocator)(size_t, void*),
            void (*deleter)(void*, size_t, void*),
            void (*copy)(void*, void*, size_t, void*),
            std::string const& domain_id,
            void* metadata=nullptr):
        metadata_ptr(metadata),
        alloc_fn(allocator),
        free_fn(deleter),
        memcpy_fn(copy),
        user_prefix(domain_id)
    {}
    void* alloc(size_t n) override {
        return alloc_fn(n, metadata_ptr);
    }
    void free(void* ptr, size_t n) override {
        free_fn(ptr, n, metadata_ptr);
    }
    void memcpy(void* dst, void* src, size_t n) override {
        memcpy_fn(dst, src, n, metadata_ptr);
    }
    bool equal(pressio_domain const& rhs) const noexcept override {
        return static_cast<const pressio_domain*>(this) == &rhs;
    }
    std::string const& prefix() const override {
        return user_prefix;
    }
    std::shared_ptr<pressio_domain> clone() override {
        return shared_from_this();
    }

    void* metadata_ptr;
    void* (*alloc_fn)(size_t n, void*);
    std::function<void(void*, size_t, void*)> free_fn;
    void (*memcpy_fn)(void*, void*, size_t, void*);
    std::string user_prefix = "user";
};
#endif /* end of include guard: PRESSIO_USER_DOMAIN_H */
