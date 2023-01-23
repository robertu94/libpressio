#ifndef LIBPRESSIO_ILESS
#define LIBPRESSIO_ILESS
#include <string>

struct iless {
    bool operator()(std::string lhs, std::string rhs) const;
};  
#endif /* end of include guard: LIBPRESSIO_ILESS */
