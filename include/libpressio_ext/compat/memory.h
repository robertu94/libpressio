/**
 * \file
 * \brief back ports of `<memory>`
 */
#ifndef LIBPRESSIO_COMPAT_MEMORY_H
#define LIBPRESSIO_COMPAT_MEMORY_H
#include <pressio_version.h>
#include <memory>
#include "type_traits.h"


namespace compat {
#if (!LIBPRESSIO_COMPAT_HAS_MAKE_UNIQUE)
  /**
   * \param args arguments to be forwarded to the constructor
   * \returns a unique_ptr with newly allocated data
   */
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
#else
    using std::make_unique;
#endif


}


#endif /* end of include guard: LIBPRESSIO_COMPAT_MEMORY_H */
