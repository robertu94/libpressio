#include "timer.h"

namespace libpressio {
namespace utils {
void start(timer& t) {
    t = time_range();
    t->begin = high_resolution_clock::now();
}
void stop(timer& t) {
    t->end = high_resolution_clock::now();
}
compat::optional<uint64_t> elapsed(timer const& t) {
    if(t) return t->elapsed();
    else return compat::nullopt;
}
}}
