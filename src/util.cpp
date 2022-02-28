#include "vo_nono/util.h"

namespace vo_nono {
    uint64_t rand64() {
        static thread_local std::random_device rd_dv;
        static thread_local std::mt19937 rd{rd_dv()};
        return rd();
    }
}
