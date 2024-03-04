#ifndef PRESISO_MEMORY_H_CXRZOVAC
#define PRESISO_MEMORY_H_CXRZOVAC
#include "domain.h"
#include <cstddef>
#include <memory>

/**
 * \file
 * \brief C++ interface to managed memory objects from domaines
 */


class pressio_memory {
private:
  std::shared_ptr<pressio_domain> data_domain;
  void *data_ptr;
  size_t data_capacity;

public:
  /**
   * copy assigns a memory object
   *
   * if the rhs will not fit in the lhs, realloc in the lhs domain
   * if the lhs and rhs are in the same domain, use the domain copy function
   * else use the pressio_copy_engine to copy between domains
   *
   * \param[in] rhs the memory to copy from
   */
  pressio_memory &operator=(pressio_memory const &rhs);
  /**
   * move assigns a memory object
   *
   * if the domains are the same, swap the pointers
   * if the rhs will not fit in the lhs, realloc in the lhs domain
   * if the lhs and rhs are in the same domain, use the domain copy function
   * else use the pressio_copy_engine to copy between domains
   *
   * \param[in] rhs the memory to move from
   */
  pressio_memory &operator=(pressio_memory &&rhs) noexcept;

  // default constructors
  /**
   * create a memory without data in the malloc domain
   */
  pressio_memory();

  /**
   * frees a pressio_memory
   */
  ~pressio_memory();
  /**
   * create a memory without data in the provided domain
   *
   * \param[in] domain the domain to create in
   */
  pressio_memory(std::shared_ptr<pressio_domain> &&domain);
  /**
   * create a memory without data in the provided domain
   *
   * \param[in] domain the domain to create in
   */
  pressio_memory(std::shared_ptr<pressio_domain> const&domain);

  // owning constructors
  /**
   * create a memory with n bytes in the malloc domain
   *
   * \param[in] n the size in bytes to allocate
   */
  pressio_memory(size_t n);
  /**
   * create a memory with n bytes in the provided domain
   *
   * \param[in] n the size in bytes to allocate
   * \param[in] domain the domain to create in
   */
  pressio_memory(size_t n, std::shared_ptr<pressio_domain> &&domain);
  /**
   * create a memory with n bytes in the provided domain
   *
   * \param[in] n the size in bytes to allocate
   * \param[in] domain the domain to create in
   */
  pressio_memory(size_t n, std::shared_ptr<pressio_domain> const&domain);

  // non-owning constructor
  /**
   * create a memory from a pre-allocated pointer in the nonowning domain
   *
   * \param[in] ptr the pointer to use
   * \param[in] n the capacity of the pointer
   */
  pressio_memory(void *ptr, size_t n);
  /**
   * create a memory from a pre-allocated pointer in the provided domain
   *
   * \param[in] ptr the pointer to use
   * \param[in] n the capacity of the pointer
   * \param[in] domain the domain of the pointer
   */
  pressio_memory(void *ptr, size_t n, std::shared_ptr<pressio_domain> &&domain);
  /**
   * create a memory from a pre-allocated pointer in the provided domain
   *
   * \param[in] ptr the pointer to use
   * \param[in] n the capacity of the pointer
   * \param[in] domain the domain of the pointer
   */
  pressio_memory(void *ptr, size_t n, std::shared_ptr<pressio_domain> const&domain);
  // copy constructor
  /**
   * copy the pointer into the malloc domain
   *
   * \param[in] rhs the memory to copy from
   */
  pressio_memory(pressio_memory const &rhs);
  /**
   * copy the pointer into the provided domain
   *
   * \param[in] rhs the memory to copy from
   * \param[in] domain the domain to copy into
   */
  pressio_memory(pressio_memory const &rhs, std::shared_ptr<pressio_domain> &&domain);
  /**
   * copy the pointer into the provided domain
   *
   * \param[in] rhs the memory to copy from
   * \param[in] domain the domain to copy into
   */
  pressio_memory(pressio_memory const &rhs, std::shared_ptr<pressio_domain> const&domain);
  // move constructor
  /**
   * move the pointer into another memory object; cloning the source domain
   *
   * \param[in] rhs the memory to move from
   */
  pressio_memory(pressio_memory &&rhs) noexcept;
  /**
   * move the pointer into another memory object using the provided domain
   *
   * \param[in] rhs the memory to copy from
   * \param[in] domain the domain to copy into
   */
  pressio_memory(pressio_memory &&rhs, std::shared_ptr<pressio_domain> &&domain);
  /**
   * move the pointer into another memory object using the provided domain
   *
   * \param[in] rhs the memory to copy from
   * \param[in] domain the domain to copy into
   */
  pressio_memory(pressio_memory &&rhs, std::shared_ptr<pressio_domain> const&domain);

  /**
   * pointer to the data in the memory
   */
  void *data() const;
  /**
   * capacity of the memory
   */
  size_t capacity() const;
  /**
   * access the domain of the memory
   */
  std::shared_ptr<pressio_domain> const& domain() const;
};

#endif /* end of include guard: MEMORY_H_CXRZOVAC */
