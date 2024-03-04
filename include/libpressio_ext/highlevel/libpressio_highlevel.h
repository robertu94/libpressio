#ifndef LIBPRESSIO_HIGHLEVEL_H_EDKS8E0P
#define LIBPRESSIO_HIGHLEVEL_H_EDKS8E0P
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/io/pressio_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file
 * \brief highlevel C interface for libpressio
 */

    /**
     * A highlevel api to create a compressor from options containing only strings
     *
     * \param[in] library       pressio object used to create the compressor
     * \param[in] compressor_id not-null; which compressor to build. 
     * \param[in] early_config  configuration options used to setup the compressor tree
     *                          its configuration is not changed to match the types provided
     *                          by the compressor. Generally you will pass an options structure
     *                          containing either/or strings or lists of strings to set the list
     *                          of meta-compressors.  null can be passed, and these will be ignored
     * \param[in] config        configuration options used to configure the compressor.
     *                          These options are casted to match the types provided by the
     *                          compressor using pressio_conversion_special.  null can be passed and
     *                          these will be ignored.
     * \returns the newly created compressor or null on an error
     */
    struct pressio_compressor*
    pressio_highlevel_get_compressor(struct pressio* library,
                                     const char* compressor_id,
                                     struct pressio_options const* early_config,
                                     struct pressio_options const* config
                                     );
                                      
    /**
     * A highlevel api to create a compressor from options containing only strings
     *
     * \param[in] library       pressio object used to create the io
     * \param[in] io_id         not-null; which io to build.
     * \param[in] early_config  configuration options used to setup the io tree
     *                          its configuration is not changed to match the types provided
     *                          by the compressor. Generally you will pass an options structure
     *                          containing either/or strings or lists of strings to set the list
     *                          of meta-io.  null can be passed, and these will be ignored
     * \param[in] config        configuration options used to configure the io.
     *                          These options are casted to match the types provided by the
     *                          compressor using pressio_conversion_special.  null can be passed and
     *                          these will be ignored.
     * \returns the newly created io or null on an error
     */
    struct pressio_io*
    pressio_highlevel_get_io(struct pressio* library,
                                     const char* io_id,
                                     struct pressio_options const* early_config,
                                     struct pressio_options const* config
                                     );
    

#ifdef __cplusplus
}
#endif

#endif /* end of include guard: LIBPRESSIO_HIGHLEVEL_H_EDKS8E0P */
