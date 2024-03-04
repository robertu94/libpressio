#include <string.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/**
 * \file
 * \brief C interface that hashes libpressio data object keys and entries.
 */

struct pressio;
struct pressio_options;

/**
 * hash the keys of a pressio_options value.  See libpressio_hash_entries for additional caveats
 *
 * \param[in] library optional; used for error reporting
 * \param[in] options the options structure to hash
 * \param[out] output_size the size of the returned hash
 * \returns memory containing the hash as bytes, should be freed with free
 */
uint8_t* libpressio_options_hashkeys(struct pressio* library, struct pressio_options const* options, size_t* output_size);

/**
 * hash the entries of a pressio_options value
 *
 * ignores the values of entries the following types:
 *   + pressio_userptr_type
 *
 * This function MAY not be consistent in the following circumstances
 *  + different architectures or compilation targets
 *  + different versions of the C or C++ standard library
 *  + different versions of OpenSSL
 *  + different versions of LibPressio
 *  + different configurations of global variables that effect formatting such as locale
 *
 *  This function SHOULD be consistent across executions
 *
 * \param[in] library optional; used for error reporting
 * \param[in] options the options structure to hash
 * \param[out] output_size the size of the returned hash
 * \returns memory containing the hash as bytes, should be freed with free
 */
uint8_t* libpressio_options_hashentries(struct pressio* library, struct pressio_options const* options, size_t* output_size);

#ifdef __cplusplus
}
#endif
