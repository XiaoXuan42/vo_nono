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
template<typename T>
[[nodiscard]] inline std::vector<T> mask_chaining(const std::vector<T> &mask1,
                                                  const std::vector<T> &mask2) {
    assert(mask1.size() >= mask2.size());
    int cur = 0;
    std::vector<T> result = mask1;
    for (int i = 0; i < (int) result.size(); ++i) {
        if (mask1[i]) {
            assert(cur < (int) mask2.size());
            result[i] = mask2[cur];
            cur += 1;
        }
    }
    return result;
}

template<typename T>
[[nodiscard]] inline std::vector<T> mask_and(const std::vector<T> &mask1, const std::vector<T> &mask2) {
    assert(mask1.size() == mask2.size());
    std::vector<T> result(mask1.size());
    for (int i = 0; i < (int) mask1.size(); ++i) {
        if (mask1[i] && mask2[i]) {
            result[i] = true;
        } else {
            result[i] = false;
        }
    }
    return result;
}
}// namespace vo_nono

#endif