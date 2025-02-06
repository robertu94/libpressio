#ifndef LIBPRESSIO_TIMER_H
#define LIBPRESSIO_TIMER_H
#include <chrono>
#include "std_compat/optional.h"

namespace libpressio {
namespace utils {
using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
struct time_range{
    time_point<high_resolution_clock> begin;
    time_point<high_resolution_clock> end;
    unsigned int elapsed() const { return duration_cast<milliseconds>(end-begin).count(); }
};
using timer = compat::optional<time_range>;
void start(timer& t);
void stop(timer& t);
compat::optional<uint64_t> elapsed(timer const& t);
}}

#endif /*LIBPRESSIO_TIMER_H*/
