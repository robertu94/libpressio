#ifndef LIBPRESSIO_PYTHON_MEMORY_HELPERS_H
#define LIBPRESSIO_PYTHON_MEMORY_HELPERS_H
#include "pressio_data.h"
#include <cstdint>

// Test-only entry points: raw handles follow the normal pressio.data_free
// contract. Moving consumes the *value*, not the handle, so both handles must
// still be freed. No counter reset is provided: it would hide retained storage.
int64_t memory_test_allocations();
int64_t memory_test_bytes();
int64_t memory_test_sends();
struct pressio_data* memory_test_new(bool alternate, bool empty);
struct pressio_data* memory_test_readable_buffer(struct pressio_data* destination,
                                               const struct pressio_data* source);
struct pressio_data* memory_test_readable_domain(bool alternate, bool move_domain,
                                               const struct pressio_data* source);
struct pressio_data* memory_test_writeable(bool alternate, bool move_source,
                                         struct pressio_data* source);
void memory_test_move_assign(struct pressio_data* destination,
                             struct pressio_data* source);
void memory_test_readable_alias_assign(struct pressio_data* data);
#endif
