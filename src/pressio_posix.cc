#include <string>
#include <cstring>
#include <algorithm>
#include <errno.h>

std::string errno_to_error() {
    // gnulibc insists on providing their implementation
    // of strerror_r when compiling with C++ which returns a char* which
    // is declared warn_unused.  However, there is also a POSIX version
    // which returns a int. BAD GNU...
    //
    // The other versions are either not thread safe or GNU extensions

    auto ec = errno;
    char err_buf[1024];
    std::fill(err_buf, err_buf+1024, '\0');
#if _GNU_SOURCE
    //assume gnu version
    char* rc = strerror_r(ec, err_buf, 1024);
    return rc;
#else
    //assume xsi version
    int rc = strerror_r(ec, err_buf, 1024);
    if(rc == 0) return std::string(err_buf);
    else return "failed to get error msg";
#endif
}
