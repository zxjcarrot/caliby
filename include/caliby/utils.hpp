#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>
#include <deque>
#include <cstring> // For memset/memcpy
#include <future>
#include <queue>
#include <sys/mman.h> // mmap

// Basic types
using uint32_t = unsigned int;

// --- A simple thread-safe set for tracking deleted nodes ---
template<typename T>
class ConcurrentSet {
private:
    std::unordered_set<T> set_;
    mutable std::shared_mutex mutex_;
public:
    void insert(const T& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        set_.insert(value);
    }
    bool contains(const T& value) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return set_.count(value) > 0;
    }
    std::vector<T> get_elements() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return std::vector<T>(set_.begin(), set_.end());
    }
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        set_.clear();
    }
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return set_.size();
    }
};

// --- VisitedList for preventing cycles during graph traversal ---
using vl_type = unsigned short;

class VisitedList {
public:
    vl_type curV;
    vl_type* mass;
    uint32_t numelements;
    VisitedList* next = nullptr;

    VisitedList(int numelements1) {
        curV = 1;
        numelements = numelements1;
        // use mmap and madvise huge page to allocate memory
        mass = static_cast<vl_type*>(mmap(nullptr, sizeof(vl_type) * numelements, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (mass == MAP_FAILED) {
            throw std::runtime_error("Failed to allocate memory for VisitedList");
        }
        madvise(mass, sizeof(vl_type) * numelements, MADV_HUGEPAGE);
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV = 1;
        }
    }

    // Check if a node is visited. If not, mark it as visited.
    // Returns true if the node was not visited before, false otherwise.
    bool try_visit(uint32_t node_id) {
        if (mass[node_id] != curV) {
            mass[node_id] = curV;
            return true;
        }
        return false;
    }

    // Explicitly un-visit a node (used for retries on I/O failure)
    void unvisit(uint32_t node_id) {
        if (mass[node_id] == curV) {
            mass[node_id] = 0;  // Or curV - 1, depending on reset logic
        }
    }
    ~VisitedList() { munmap(mass, sizeof(vl_type) * numelements); }
};

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    int numelements;
    // Single cached VisitedList per thread with RAII cleanup on thread exit
    struct TLSCache {
        VisitedList* ptr = nullptr;
        ~TLSCache() { delete ptr; }
    };
    static thread_local TLSCache tls_cache;

   public:
    VisitedListPool(int /*initmaxpools*/, int numelements1) : numelements(numelements1) {}

    VisitedList* getFreeVisitedList() {
        VisitedList*& cached = tls_cache.ptr;
        if (cached) {
            if (cached->numelements == static_cast<unsigned int>(numelements)) {
                VisitedList* rez = cached;
                cached = nullptr;
                rez->next = nullptr;
                rez->reset();
                return rez;
            } else {
                // Cached list has different size; free it so we can allocate correct one
                delete cached;
                cached = nullptr;
            }
        }
        // None available for this thread, allocate a new one with correct size
        VisitedList* rez = new VisitedList(numelements);
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList* vl) {
        VisitedList*& cached = tls_cache.ptr;
        // If we already have a cached one, delete it to avoid leaks (keep single cached instance)
        if (cached && cached != vl) {
            delete cached;
            cached = nullptr;
        }
        // Only cache if dimensions match this pool's configuration; otherwise free
        if (vl && vl->numelements == static_cast<unsigned int>(numelements)) {
            cached = vl;
        } else if (vl) {
            delete vl;
        }
    }

    // No dynamic resources owned by pool itself
    ~VisitedListPool() = default;
};

// Definition of the thread_local cache (inline to avoid multiple definitions across TUs)
inline thread_local VisitedListPool::TLSCache VisitedListPool::tls_cache{};

class ThreadPool {
   public:
    ThreadPool(size_t threads) : stop(false) {
        // Create the specified number of worker threads.
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                // The main loop for each worker thread.
                for (;;) {
                    std::function<void()> task;
                    {
                        // Acquire a lock on the task queue.
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        
                        // Wait until there's a task or the pool is stopped.
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        
                        // If the pool is stopped and the queue is empty, the thread can exit.
                        if (this->stop && this->tasks.empty()) return;
                        
                        // Get the next task from the queue.
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    } // Lock is released here.
                    
                    // Execute the task.
                    task();
                }
            });
        }
    }

    // Enqueues a task to be executed by the thread pool.
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        // Create a packaged_task to wrap the function, which allows us to get a future.
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            
        // Get the future associated with the task. The caller can use this to wait for the result.
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // Don't allow enqueueing after stopping the pool.
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

            // Add the task to the queue.
            tasks.emplace([task]() { (*task)(); });
        }
        
        // Notify one waiting worker thread that a new task is available.
        condition.notify_one();
        return res;
    }
    
    // Returns the number of worker threads in the pool.
    size_t getThreadCount() const {
        return workers.size();
    }

    // Waits for all tasks currently in the queue to be completed.
    void waitForTasks() {
        // Create a promise and get its future. This future will be used to wait for the signal.
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(tasks.empty()) {
                // If the queue is already empty, no need to wait.
                return;
            }
            // Enqueue a "barrier" task. This task's only job is to signal the promise.
            // Because the queue is FIFO, this task will only execute after all previously
            // enqueued tasks have been completed.
            tasks.emplace([&promise]() { promise.set_value(); });
        }
        
        // Notify a worker to process the queue, which will eventually process our barrier task.
        condition.notify_one();
        
        // Wait for the future to be set by the barrier task.
        future.wait();
    }


    // The destructor joins all threads.
    ~ThreadPool() {
        {
            // Acquire lock and set the stop flag.
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        // Notify all waiting threads to wake up.
        condition.notify_all();
        
        // Wait for all worker threads to finish their current task and exit.
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

   private:
    std::vector<std::thread> workers;           // The worker threads.
    std::queue<std::function<void()>> tasks;    // The task queue.
    std::mutex queue_mutex;                     // Mutex to protect the task queue.
    std::condition_variable condition;          // Condition variable for synchronization.
    bool stop;                                  // Flag to stop the thread pool.
};

#endif // UTILS_HPP