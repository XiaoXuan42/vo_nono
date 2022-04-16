#ifndef VO_NONO_QUEUE_H
#define VO_NONO_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>

namespace vo_nono {
template<typename T>
class ConsumerBlockingQueue {
public:
    ConsumerBlockingQueue() = default;
    ConsumerBlockingQueue(const ConsumerBlockingQueue &) = delete;
    ConsumerBlockingQueue(ConsumerBlockingQueue &&other) noexcept {
        std::unique_lock<std::mutex> lk1(other.mutex_, mutex_);
        q_ = std::move(other.q_);
    }

    void push(const T &val) {
        std::unique_lock<std::mutex> lk(mutex_);
        q_.push(val);
        cv.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lk(mutex_);
        cv.wait(lk, [this] { return !this->q_.empty(); });
        T res = q_.front();
        q_.pop();
        return res;
    }

private:
    [[nodiscard]] bool empty() const { return q_.empty(); }
    std::queue<T> q_;
    std::mutex mutex_;
    std::condition_variable cv;
};
}// namespace vo_nono

#endif//VO_NONO_QUEUE_H
