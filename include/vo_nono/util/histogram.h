#ifndef VO_NONO_HISTOGRAM_H
#define VO_NONO_HISTOGRAM_H

namespace vo_nono {
template<typename T>
class Histogram {
public:
    Histogram(int N, std::function<int(T)> indexer)
        : indexer_(indexer),
          boxes_(N, 0),
          topK_(0) {}

    void insert_element(T val) {
        int index = indexer_(val);
        assert(index >= 0 && index < (int) boxes_.size());
        boxes_[index] += 1;
    }

    void cal_topK(int k) {
        topK_ = -1;
        std::vector<int> buffer(k, -1);
        for (auto val : boxes_) {
            if (val > buffer[k - 1]) { buffer[k - 1] = val; }
            int cur = k - 1;
            while (cur > 0) {
                if (buffer[cur] > buffer[cur - 1]) {
                    int tmp = buffer[cur];
                    buffer[cur] = buffer[cur - 1];
                    buffer[cur - 1] = tmp;
                    cur -= 1;
                } else {
                    break;
                }
            }
        }
        topK_ = buffer[k - 1];
    }

    bool is_topK(T val) const {
        int index = indexer_(val);
        assert(index >= 0 && index < (int) boxes_.size());
        return boxes_[index] >= topK_;
    }

private:
    std::function<int(T)> indexer_;
    std::vector<int> boxes_;
    int topK_;
};

}// namespace vo_nono
#endif//VO_NONO_HISTOGRAM_H
