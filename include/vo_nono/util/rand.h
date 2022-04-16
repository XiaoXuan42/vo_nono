#ifndef VO_NONO_RAND_H
#define VO_NONO_RAND_H

namespace vo_nono {
class Rand {
public:
    static uint64_t rand64() {
        static thread_local std::random_device rd_dv;
        static thread_local std::mt19937 rd{rd_dv()};
        return rd();
    }
};
}
#endif//VO_NONO_RAND_H
