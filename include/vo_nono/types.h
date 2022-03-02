#ifndef VO_NONO_TYPES_H
#define VO_NONO_TYPES_H

#include <chrono>
#include <cstdint>
#include <memory>

namespace vo_nono {
using vo_time_t = std::chrono::time_point<std::chrono::system_clock,
                                          std::chrono::milliseconds>;

inline vo_time_t vo_time_from(double t) {
    std::chrono::milliseconds dur((int64_t) (t * 1000));
    return std::chrono::time_point<std::chrono::system_clock,
                                   std::chrono::milliseconds>(dur);
}

using vo_id_t = int64_t;

template<typename T>
using vo_ptr = std::shared_ptr<T>;
template<typename T>
using vo_uptr = std::unique_ptr<T>;
}// namespace vo_nono

#endif//VO_NONO_TYPES_H
