#ifdef __cplusplus
extern "C" {
#endif


#ifndef LIBPRESSIO_OPTIONS_ITER
#define LIBPRESSIO_OPTIONS_ITER
#include <stdbool.h>

/*!
 * \file
 * \brief An iterator for a set of options for a compressor
 */

struct pressio_options;
struct pressio_options_iter;

/**
 * Get an iterator over all keys
 * the use of any of the pressio_options_set* functions invalidates the iterator
 *
 * \returns an iterator over all keys
 */
struct pressio_options_iter* pressio_options_get_iter(struct pressio_options const* options);
/**
 * Returns true if the current position has a value
 * \param[in] iter to check if it has a next value
 * \returns true if the current iterator position has a value
 */
bool pressio_options_iter_has_value(struct pressio_options_iter* iter);
/**
 * advances the iterator to the next entry
 * \param[in] iter the iterator to advance
 */
void pressio_options_iter_next(struct pressio_options_iter* iter);
/**
 * get the key for the current position of the iterator
 * \param[in] iter the iterator to get the current value for
 * \returns a non-owning pointer to the key name associated with the current position of the iterator
 */
char const* pressio_options_iter_get_key(struct pressio_options_iter* const iter);
/**
 * get the value for the current position of the iterator
 * \param[in] iter the iterator to get the current value for
 * \returns a owning pointer to the value associated with the current position of the iterator
 */
struct pressio_option * pressio_options_iter_get_value(struct pressio_options_iter* const iter);
/**
 * \param[in] iter the iterator to free
 * frees memory associated with the iterator
 */
void pressio_options_iter_free(struct pressio_options_iter* const iter);
#endif

#ifdef __cplusplus
}
#endif
