#ifndef PRESSIO_USER_DOMAIN_H
#define PRESSIO_USER_DOMAIN_H
#include "libpressio_ext/cpp/domain.h"
#include "pressio_data.h"
#include "libpressio_ext/cpp/data.h"
#include "pressio_version.h"
#include <functional>

/**
 * \file
 * \brief domain that uses user-defined C allocation and free functions
 */
#if LIBPRESSIO_HAS_CUDA
extern "C" {
    void* pressio_cuda_alloc_fn(size_t n, void*);
    void* pressio_cudahost_alloc_fn(size_t n, void*);
    void  pressio_cuda_memcpy_fn(void*, void*, size_t, void*);
}
#endif
extern "C" {
    void* pressio_noop_alloc_fn(size_t n, void*);
    void* pressio_libc_alloc_fn(size_t n, void*);
    void pressio_libc_memcpy_fn(void* dst, void* src, size_t n, void*);
}

namespace impl {
}

struct pressio_user_domain: public pressio_domain, std::enable_shared_from_this<pressio_user_domain> {
    pressio_user_domain(void (*deleter)(void*, void*), void* metadata=nullptr, std::vector<std::string> const& accessible={}):
        metadata_ptr(metadata),
        accessible(accessible)
    {
        if(deleter) {
            //special case this because it is so common
            if(deleter == pressio_data_libc_free_fn && accessible.empty()) {
                this->accessible = std::vector<std::string>{};
                this->user_prefix = "malloc";
                this->alloc_fn = pressio_libc_alloc_fn;
                this->memcpy_fn = pressio_libc_memcpy_fn;
            }
            else if(is_pressio_new_free_fn(deleter) && accessible.empty()) {
                this->accessible = std::vector<std::string>{};
                this->user_prefix = "malloc";
                this->alloc_fn = pressio_libc_alloc_fn;
                this->memcpy_fn = pressio_libc_memcpy_fn;

            }
#if LIBPRESSIO_HAS_CUDA
            else if(deleter == pressio_data_cuda_free_fn && accessible.empty())  {
                this->accessible = std::vector<std::string>{""};
                this->user_prefix = "cudamalloc";
                this->alloc_fn = pressio_cuda_alloc_fn;
                this->memcpy_fn = pressio_cuda_memcpy_fn;
            } else if(deleter == pressio_data_cuda_free_fn && accessible.empty()) {
                this->accessible = std::vector<std::string>{"malloc"};
                this->user_prefix = "cudamallochost";
                this->alloc_fn = pressio_cudahost_alloc_fn;
                this->memcpy_fn = pressio_libc_memcpy_fn;
            }
#endif
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
            void* metadata=nullptr,
            std::vector<std::string> const& accessible={}
            ):
        metadata_ptr(metadata),
        alloc_fn(allocator),
        free_fn(deleter),
        memcpy_fn(copy),
        user_prefix(domain_id),
        accessible(accessible)
    {}
    domain_options get_configuration_impl() const override {
        domain_options opts;
        set(opts, get_name(), "domains:accessible", accessible);
        return opts;
    }
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

    void* metadata_ptr = nullptr;
    void* (*alloc_fn)(size_t n, void*) = pressio_noop_alloc_fn;
    std::function<void(void*, size_t, void*)> free_fn;
    void (*memcpy_fn)(void*, void*, size_t, void*) = pressio_libc_memcpy_fn;
    std::string user_prefix = "user";
    std::vector<std::string> accessible;
};
#endif /* end of include guard: PRESSIO_USER_DOMAIN_H */
