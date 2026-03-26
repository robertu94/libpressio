# static libraries and some system linkers will strip the registration classes 
# causing nothing to be registered
message("STATUS" "LIBPRESSIO_BUILT_PLUGINS = ${LIBPRESSIO_BUILT_PLUGINS}")

# extract the plugin types and names and sort them
list(TRANSFORM LIBPRESSIO_BUILT_PLUGINS REPLACE ".*\\/([a-zA-Z0-9_]*)\\/([a-zA-Z0-9_]*)\.[a-zA-Z0-9_]*\$" "\\1/\\2" OUTPUT_VARIABLE LIBPRESSIO_REGISTRATION_PLUGINS)
list(SORT LIBPRESSIO_REGISTRATION_PLUGINS)

# compute the per-plugin extern registrations
list(TRANSFORM LIBPRESSIO_REGISTRATION_PLUGINS REPLACE "([a-zA-Z0-9_]*)\\/([a-zA-Z0-9_]*)\$" "  namespace \\1 { namespace \\2_ns { extern pressio_register registration! } }" OUTPUT_VARIABLE LIBPRESSIO_REGISTRATION_EXTERNS)
string(REPLACE ";" "\n" LIBPRESSIO_REGISTRATION_EXTERNS "${LIBPRESSIO_REGISTRATION_EXTERNS}")
string(REPLACE "!" ";" LIBPRESSIO_REGISTRATION_EXTERNS "${LIBPRESSIO_REGISTRATION_EXTERNS}")

# compute the per-plugin registration calls
list(TRANSFORM LIBPRESSIO_REGISTRATION_PLUGINS REPLACE "([a-zA-Z0-9_]*)\\/([a-zA-Z0-9_]*)\$" "  libpressio::\\1::\\2_ns::registration.ensure_registered()!" OUTPUT_VARIABLE LIBPRESSIO_REGISTRATION_CALLS)
string(REPLACE ";" "\n" LIBPRESSIO_REGISTRATION_CALLS "${LIBPRESSIO_REGISTRATION_CALLS}")
string(REPLACE "!" ";" LIBPRESSIO_REGISTRATION_CALLS "${LIBPRESSIO_REGISTRATION_CALLS}")

# interpolate the manual plugin registration code
set(LIBPRESSIO_REGISTRATIONS "
#include <iostream>
#include \"libpressio_ext/cpp/registry.h\"

namespace libpressio {
${LIBPRESSIO_REGISTRATION_EXTERNS}
} /*namespace libpressio*/

extern \"C\" void pressio_register_all() {
${LIBPRESSIO_REGISTRATION_CALLS}
}
")

# write out the plugin registration code
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pressio_manual_register.cc "${LIBPRESSIO_REGISTRATIONS}")
