#include "vo_nono/map.h"

#include "vo_nono/optimize_graph.h"

namespace vo_nono {
void Map::global_bundle_adjustment() {
    std::unique_lock<std::mutex> glb_lock(map_global_mutex);
    while (true) {
        if (mb_shutdown) { break; }
        m_global_ba_cv.wait(glb_lock, [&] { return mb_global_ba; });
        if (mb_shutdown) { break; }
        mb_global_ba = false;
        if (m_keyframes.size() < 5) { continue; }

        _global_bundle_adjustment(glb_lock);
    }
}

void Map::_global_bundle_adjustment(std::unique_lock<std::mutex> &lock) {
}
}// namespace vo_nono