#include <memory>
#include <string>
#include <std_compat/shared_mutex.h>
#include <std_compat/mutex.h>
#include <pressio_dtype.h>
#include <sz/sz.h>

#define PRESSIO_SZ_VERSION_GREATEREQ(major, minor, build, revision) \
   (SZ_VER_MAJOR > major || \
   (SZ_VER_MAJOR == major && SZ_VER_MINOR > minor) ||                                  \
   (SZ_VER_MAJOR == major && SZ_VER_MINOR == minor && SZ_VER_BUILD > build) || \
   (SZ_VER_MAJOR == major && SZ_VER_MINOR == minor && SZ_VER_BUILD == build && SZ_VER_REVISION >= revision))

namespace libpressio {
int libpressio_type_to_sz_type(pressio_dtype type);

struct sz_init_handle {
  sz_init_handle();
  ~sz_init_handle();
  sz_init_handle(sz_init_handle const&)=delete;
  sz_init_handle& operator=(sz_init_handle const&)=delete;
  sz_init_handle(sz_init_handle &&)=delete;
  sz_init_handle& operator=(sz_init_handle &&)=delete;

  
  compat::shared_mutex sz_init_lock;
};

std::shared_ptr<sz_init_handle> pressio_get_sz_init_handle();
}
