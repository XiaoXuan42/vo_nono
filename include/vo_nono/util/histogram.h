#ifndef VO_NONO_HISTOGRAM_H
#define VO_NONO_HISTOGRAM_H

namespace vo_nono {
template<typename T>
class Histogram {
public:
    Histogram(int N, std::function<int(T)> indexer)
            : m_indexer(indexer),
              m_boxes(N, 0),
              m_topK(0) {}

    void insert_element(T val) {
        int index = m_indexer(val);
        assert(index >= 0 && index < (int) m_boxes.size());
        m_boxes[index] += 1;
    }

    void cal_topK(int k) {
        m_topK = -1;
        std::vector<int> buffer(k, -1);
        for (auto val : m_boxes) {
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
        m_topK = buffer[k - 1];
    }

    bool is_topK(T val) const {
        int index = m_indexer(val);
        assert(index >= 0 && index < (int) m_boxes.size());
        return m_boxes[index] >= m_topK;
    }

private:
    std::function<int(T)> m_indexer;
    std::vector<int> m_boxes;
    int m_topK;
};

}
#endif//VO_NONO_HISTOGRAM_H
