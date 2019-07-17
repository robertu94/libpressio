#ifdef __cplusplus
extern "C" {
#endif


#ifndef LIBLOSSY_OPTIONS_ITER
#define LIBLOSSY_OPTIONS_ITER
#include <stdbool.h>

/*!
 * \file
 * \brief An iterator for a set of options for a compressor
 */

struct lossy_options;
struct lossy_options_iter;

/**
 * Get an iterator over all keys
 * the use of any of the lossy_options_set* functions invalidates the iterator
 *
 * \returns an iterator over all keys
 */
struct lossy_options_iter* lossy_options_get_iter(struct lossy_options const* options);
/**
 * Returns true if the current position has a value
 * \param[in] iter to check if it has a next value
 * \returns true if the current iterator position has a value
 */
bool lossy_options_iter_has_value(struct lossy_options_iter* iter);
/**
 * advances the iterator to the next entry
 * \param[in] iter the iterator to advance
 */
void lossy_options_iter_next(struct lossy_options_iter* iter);
/**
 * get the key for the current position of the iterator
 * \param[in] iter the iterator to get the current value for
 * \returns a non-owning pointer to the key name associated with the current position of the iterator
 */
char const* lossy_options_iter_get_key(struct lossy_options_iter* const iter);
/**
 * get the value for the current position of the iterator
 * \param[in] iter the iterator to get the current value for
 * \returns a owning pointer to the value associated with the current position of the iterator
 */
struct lossy_option * lossy_options_iter_get_value(struct lossy_options_iter* const iter);
/**
 * \param[in] iter the iterator to free
 * frees memory associated with the iterator
 */
void lossy_options_iter_free(struct lossy_options_iter* const iter);
#endif

#ifdef __cplusplus
}
#endif
