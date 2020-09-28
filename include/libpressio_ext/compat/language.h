#include <utility>

#if __cplusplus >= 201703L
#define  RVO_MOVE(x) (x)
#define  DEFAULTED_NOEXCEPT noexcept
#else
#define  RVO_MOVE(x) std::move(x)
#define  DEFAULTED_NOEXCEPT
#endif
