#ifdef __cplusplus
extern "C" {
#endif

#ifndef PRESSIO_IO_H
#define PRESSIO_IO_H

/**
 * \file
 * \brief a generic io interface for libpressio
 */

struct pressio_io;
struct pressio_data;
struct pressio_options;


/**
 * \param[in] library the name of the io module
 * \param[in] io_module the name of the io module
 * \returns a new handle to an io module
 */
struct pressio_io* pressio_get_io(struct pressio* library, const char* io_module);

/*!
 * \param[in] io deallocates a reference to a io module.
 */
void pressio_io_free(struct pressio_io* io);

/**
 * \returns the list supported io_modules seperated by whitespace
 */
const char* pressio_supported_io_modules();

//option getting/setting functions
/*!
 * \returns a pressio options struct that represents the compile time configuration of the io module
 * \param[in] io which io to get compile-time configuration for
 */
struct pressio_options* pressio_io_get_configuration(struct pressio_io const* io);

/*!
 * \returns a pressio options struct that represents the current options of the io module
 * \param[in] io which io to get options for
 */
struct pressio_options* pressio_io_get_options(struct pressio_io const* io);
/*!
 * sets the options for the pressio_io.  io modules MAY choose to ignore
 * some subset of options passed in if there valid but conflicting settings
 * (i.e. two settings that adjust the same underlying configuration).
 * io modules SHOULD return an error value if configuration
 * failed due to a missing required setting or an invalid one. Users MAY
 * call pressio_io_error_msg() to get more information about the warnings or errors
 * io modules MUST ignore any and all options set that they do not support.
 *
 * \param[in] io which io to get options for
 * \param[in] options the options to set
 * \returns 0 if successful, positive values on errors, negative values on warnings
 *
 * \see pressio_io_error_msg
 */
int pressio_io_set_options(struct pressio_io* io, struct pressio_options const * options);
/*!
 * Validates that only defined options have been set.  This can be useful for programmer errors.
 * This function should NOT be used with any option structure which contains options for multiple io modules.
 * Other checks MAY be preformed implementing io modules.
 * 
 * \param[in] io which io to validate the options struct against
 * \param[in] options which options set to check against.  It should ONLY contain options returned by pressio_io_get_options
 * \returns 0 if successful, 1 if there is an error.  On error, an error message is set in pressio_io_error_msg.
 * \see pressio_io_error_msg to get the error message
 */
int pressio_io_check_options(struct pressio_io* io, struct pressio_options const * options);

/** reads a pressio_data buffer from some persistent storage
 * \param[in] io the object to preform the read
 * \param[in] data data object to populate, or nullptr to allocate it from the file if supported by the plugin.  Data 
 *  passed to this call should be considered "moved" in a c++11 sense.
 * \returns nullptr on failures, non-null on success
 */
struct pressio_data* pressio_io_read(struct pressio_io* io, struct pressio_data* data);

/** write a pressio_data buffer from some persistent storage
 * \param[in] io the object to preform the write
 * \param[in] data data object to write
 * \returns 0 if successful, positive values on errors, negative values on warnings
 */
int pressio_io_write(struct pressio_io* io, struct pressio_data const* data);

/**
 * \param[in] io the io module to query
 * \returns last error code for the io
 */
int pressio_io_error_code(struct pressio_io const* io);

/**
 * \param[in] io the io module to query
 * \returns last error message for the io
 */
const char* pressio_io_error_msg(struct pressio_io const* io);

/*!
 * Get the version and feature information.  The version string may include more information than major/minor/patch provide.
 * \param[in] io the io to query
 * \returns a implementation specific version string
 */
const char* pressio_io_version(struct pressio_io const* io);
/*!
 * \param[in] io the io module to query
 * \returns the major version number or 0 if there is none
 */
int pressio_io_major_version(struct pressio_io const* io);
/*!
 * \param[in] io the io module to query
 * \returns the major version number or 0 if there is none
 */
int pressio_io_minor_version(struct pressio_io const* io);
/*!
 * \param[in] io the io module to query
 * \returns the major version number or 0 if there is none
 */
int pressio_io_patch_version(struct pressio_io const* io);

#endif /* end of include guard: PRESSIO_IO_H */

#ifdef __cplusplus
}
#endif
