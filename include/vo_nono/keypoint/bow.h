#ifndef VO_NONO_BOW_H
#define VO_NONO_BOW_H

#include <DBoW3.h>

#include <unordered_map>
#include <unordered_set>

#include "vo_nono/types.h"

namespace vo_nono {
using Bow = DBoW3::BowVector;
class BowDataBase {
public:
    explicit BowDataBase(const char *filename)
        : voc_(filename),
          database_(voc_) {}
    Bow get_bow(const cv::Mat &features) {
        Bow result;
        database_.getVocabulary()->transform(features, result);
        return result;
    }
    void add(vo_id_t id, const cv::Mat &features) {
        auto entry_id = database_.add(features);
        map_[entry_id] = id;
    }
    std::vector<std::pair<vo_id_t, double>> query(const Bow &bow,
                                                  int max_results = -1,
                                                  int max_id = -1) {
        DBoW3::QueryResults results;
        database_.query(bow, results, max_results, max_id);
        std::sort(results.begin(), results.end(),
                  [](DBoW3::Result &result1, DBoW3::Result &result2) {
                      return result2 < result1;
                  });
        std::vector<std::pair<vo_id_t, double>> ret;
        for (auto &result : results) {
            vo_id_t frame_id = map_[result.Id];
            double score = result.Score;
            ret.emplace_back(frame_id, score);
        }
        return ret;
    }

private:
    DBoW3::Vocabulary voc_;
    DBoW3::Database database_;
    std::unordered_map<DBoW3::EntryId, vo_id_t> map_;
};
}// namespace vo_nono

#endif//VO_NONO_BOW_H
