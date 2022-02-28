#ifndef VO_NONO_TYPES_H
#define VO_NONO_TYPES_H

#include <chrono>

namespace vo_nono {
using vo_time_t = std::chrono::time_point<std::chrono::system_clock,
                                          std::chrono::milliseconds>;
}

#endif//VO_NONO_TYPES_H
