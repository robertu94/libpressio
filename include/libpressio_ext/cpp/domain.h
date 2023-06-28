#ifndef PRESSIO_DOMAIN_H_Z2ALCUZG
#define PRESSIO_DOMAIN_H_Z2ALCUZG
#include <cstddef>
#include <string>
#include <memory>
#include <cstring>
#include <sstream>

struct pressio_domain {
    /**
     * allocate memory in the domain
     *
     * an implementation MUST provide at least this many bytes writable from the devices that are accessible from this domain
     * an implementation MUST ensure that all allocated regions do not overlap
     * an implementation SHOULD avoid wastefully large over-allocations
     * an implementation MAY throw an exception if allocation fails
     *
     * \param[in] n minimum amount of memory in bytes to allocate
     */
    virtual void* alloc(size_t n)=0;
    /**
     * frees memory associated from the domain
     *
     * users MUST provide the same capacity used to the call to alloc
     * users MUST NOT utilize a pointer after it has been passed to free until it is returned from a subsequent alloc operation
     * an implementation SHOULD return the underlying allocation to the device
     *
     * \param[in] capacity the amount of the memory to return to the allocator
     */
    virtual void free(void* data, size_t capacity)=0;
    /**
     * moves memory within a domain
     *
     * users MUST provided a src and dst pointer from a domain that is equal to this domain
     * users MUST provide a capacity that does not exceed the capacity of the smaller of the src or dst
     *
     * \param[out] dst where to copy the data to
     * \param[in] src where to copy the data from
     * \param[in] capacity the number of bytes to copy
     */
    virtual void memcpy(void* dst, void* src, size_t capacity)=0;
    /**
     * a unique identifier for all domains that compare equal to this domain
     */
    virtual std::string prefix() const=0;
    /**
     * \returns true if and only if pointers allocated with this can be freed by rhs or visa-versa
     */
    bool operator==(pressio_domain const& rhs) const noexcept {
        return equal(rhs);
    }
    /**
     * clones a domain
     *
     * an implementation MUST preform allocations on the same device
     * an implementation SHOULD return a domain that compares equal to this domain
     */
    virtual std::shared_ptr<pressio_domain> clone()=0;

    protected:
    virtual bool equal(pressio_domain const&) const noexcept=0;
};


#endif /* end of include guard: PRESSIO_DOMAIN_H_Z2ALCUZG */
