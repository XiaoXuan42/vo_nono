#ifndef VO_NONO_UTIL_H
#define VO_NONO_UTIL_H

#include <cstdlib>
#include <string>
#include <vector>

namespace vo_nono {
template<typename T, typename U>
[[nodiscard]] inline std::vector<T> filter_by_mask(
        const std::vector<T> &targets, const std::vector<U> &mask) {
    assert(targets.size() == mask.size());
    std::vector<T> res;
    res.reserve(targets.size());
    for (int i = 0; i < (int) targets.size(); ++i) {
        if (mask[i]) { res.push_back(targets[i]); }
    }
    return res;
}
template<typename T>
[[nodiscard]] inline int cnt_inliers_from_mask(const std::vector<T> &mask) {
    int cnt = 0;
    for (int i = 0; i < (int) mask.size(); ++i) {
        if (mask[i]) { cnt += 1; }
    }
    return cnt;
}
}// namespace vo_nono

#endif