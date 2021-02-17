#include <H5PLextern.h>

#ifdef __cplusplus
extern "C" {
#endif

//a made up unique hdf5 filter value
//TODO replace with an actual value
#define H5Z_FILTER_LIBPRESSIO 21023

H5PL_type_t H5PLget_plugin_type();

herr_t H5Pset_libpressio(hid_t dcpl, const char* compressor_id, struct pressio_options const* options);

const void* H5PLget_plugin_info();

extern const H5Z_class2_t H5Z_LIBPRESSIO[1];

#ifdef __cplusplus
}
#endif
