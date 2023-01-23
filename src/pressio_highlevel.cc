#include <libpressio_ext/highlevel/libpressio_highlevel.h>
#include <libpressio_ext/cpp/libpressio.h>
#include <memory>

extern "C" {

    struct pressio_compressor*
    pressio_highlevel_get_compressor(struct pressio* library,
                                     const char* compressor_id,
                                     struct pressio_options const* early_config,
                                     struct pressio_options const* config
                                     ) {
        library->set_error(0, "");
        auto compressor = library->get_compressor(compressor_id);
        if(!compressor) {
            return nullptr;
        }
        int rc = compressor->cast_options(*early_config, *config);
        if(rc) {
            library->set_error(compressor->error_code(), compressor->error_msg());
            return nullptr;
        } else {
            return new pressio_compressor(std::move(compressor));
        }
    }

    struct pressio_io*
    pressio_highlevel_get_io(struct pressio* library,
                                     const char* io_id,
                                     struct pressio_options const* early_config,
                                     struct pressio_options const* config
                                     ) {
        library->set_error(0, "");
        auto io = library->get_io(io_id);
        if(!io) {
            return nullptr;
        }
        int rc = io->cast_options(*early_config, *config);
        if(rc) {
            library->set_error(io->error_code(), io->error_msg());
            return nullptr;
        } else {
            return new pressio_io(std::move(io));
        }
    }
}
