#ifndef VO_NONO_TYPES_H
#define VO_NONO_TYPES_H

#include <chrono>
#include <cstdint>
#include <memory>

namespace vo_nono {
using vo_id_t = int64_t;
constexpr vo_id_t vo_id_invalid = -1;

template<typename T>
using vo_ptr = std::shared_ptr<T>;
template<typename T>
using vo_uptr = std::unique_ptr<T>;
template<typename T>
using vo_wptr = std::weak_ptr<T>;
}// namespace vo_nono

#endif//VO_NONO_TYPES_H
