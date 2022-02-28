#ifndef VO_NONO_TYPES_H
#define VO_NONO_TYPES_H

#include <chrono>
#include <cstdint>
#include <memory>

namespace vo_nono {
using vo_time_t = std::chrono::time_point<std::chrono::system_clock,
                                          std::chrono::milliseconds>;
using vo_id_t = int64_t;

template<typename T>
using vo_ptr = std::shared_ptr<T>;
template<typename T>
using vo_uptr = std::unique_ptr<T>;
}

#endif//VO_NONO_TYPES_H
