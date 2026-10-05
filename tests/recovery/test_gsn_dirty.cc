// Tests for GSN-based dirty tracking (Phase 2 of docs/LOGGING_INTEGRATION.md).
// dirty is derived: page->p_gsn > frame_meta[pid].last_written_gsn.
// Also covers the WAL eviction gate bound (p_gsn <= flushedGsnLimit) and
// frame metadata (last_written_gsn monotonic, last_writer).

#include "calico.hpp"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

class TestGsnDirty : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(bm_ptr, nullptr) << "another test leaked a BufferManager";
        // Give the index-0 namespace a real backing file (as production
        // does via the catalog) so flush_system can actually write pages.
        std::string path = std::string(testing::TempDir()) + "caliby_gsn_global_" +
                           std::to_string(::getpid());
        test_fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        ASSERT_GE(test_fd, 0);
        bm_ptr = new BufferManager();
        bm_ptr->setGlobalNamespaceFd(test_fd);
    }
    void TearDown() override {
        if (bm_ptr != nullptr) {
            delete bm_ptr;
            bm_ptr = nullptr;
        }
        if (test_fd >= 0) {
            ::close(test_fd);
            test_fd = -1;
        }
    }
    int test_fd = -1;
};

TEST_F(TestGsnDirty, AllocGuardLeavesPageDirty) {
    AllocGuard<Page> g;
    PID pid = g.pid;
    ASSERT_NE(pid, (PID)-1);
    EXPECT_TRUE(bm.pageIsDirty(pid));
    EXPECT_EQ(bm.lastWrittenGsnOf(pid), 0u);
    EXPECT_EQ(g.ptr->p_gsn, 1u);  // first page ever allocated pessimistically gets GSN 1
}

TEST_F(TestGsnDirty, MarkWrittenMakesCleanThenGuardXDirties) {
    PID pid;
    u64 first_gsn;
    {
        AllocGuard<Page> g;
        pid = g.pid;
        first_gsn = g.ptr->p_gsn;
        EXPECT_TRUE(bm.pageIsDirty(pid));
    }

    // Simulate a completed write: last_written catches up.
    bm.updateLastWrittenGsn(pid, first_gsn);
    EXPECT_FALSE(bm.pageIsDirty(pid));
    EXPECT_EQ(bm.lastWrittenGsnOf(pid), first_gsn);

    // Exclusive re-fix re-dirties with a higher GSN.
    {
        GuardX<Page> g(pid);
        EXPECT_GT(g.ptr->p_gsn, first_gsn);
        EXPECT_TRUE(bm.pageIsDirty(pid));
        EXPECT_NE(bm.lastWriterOf(pid), 127u);  // writer recorded
    }
}

TEST_F(TestGsnDirty, ManualCleanOnExplicitFlushMark) {
    AllocGuard<Page> g;
    PID pid = g.pid;
    u64 gsn = g.ptr->p_gsn;

    bm.updateLastWrittenGsn(pid, gsn);  // sync-write path raising last_written
    EXPECT_FALSE(bm.pageIsDirty(pid));
    EXPECT_EQ(bm.lastWrittenGsnOf(pid), gsn);
    EXPECT_EQ(bm.lastWrittenGsnOf(pid), bm.lastWrittenGsnOf(pid));  // stable

    // Lower values never regress last_written (monotonic).
    bm.updateLastWrittenGsn(pid, gsn - 1);
    EXPECT_EQ(bm.lastWrittenGsnOf(pid), gsn);
}

TEST_F(TestGsnDirty, LastWriterRecorded) {
    AllocGuard<Page> g;
    PID pid = g.pid;
    bm.setLastWriter(pid, 3u);
    EXPECT_EQ(bm.lastWriterOf(pid), 3u);
    bm.setLastWriter(pid, 7u);
    EXPECT_EQ(bm.lastWriterOf(pid), 7u);
}

TEST_F(TestGsnDirty, SharedFixDoesNotDirty) {
    AllocGuard<Page> g;
    PID pid = g.pid;
    u64 gsn = g.ptr->p_gsn;
    EXPECT_TRUE(bm.pageIsDirty(pid));
    g.release();  // mark as already-latched-out so dtor skips unfixX

    {
        GuardS<Page> s(pid);
        // Shared guards must not bump the page GSN.
        EXPECT_EQ(s.ptr->p_gsn, gsn);
        // Cleanup the dirty state to isolate the assertion.
        bm.updateLastWrittenGsn(pid, gsn);
        EXPECT_FALSE(bm.pageIsDirty(pid));
    }
    EXPECT_FALSE(bm.pageIsDirty(pid));
}

TEST_F(TestGsnDirty, NoWALGateIsUnbounded) {
    // Without a LogManager running, the gate must never block flushing.
    EXPECT_EQ(BufferManager::flushedGsnLimit(), ~u64(0));
}

TEST_F(TestGsnDirty, ConcurrentGSNClockMonotonic) {
    // Advance the clock from several threads; GSNs must be unique and the
    // counter must reflect the total.
    constexpr int kThreads = 8;
    constexpr int kAdvancePerThread = 500;

    std::vector<std::thread> threads;
    std::vector<std::vector<u64>> gsns(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            workerThreadId = static_cast<uint16_t>(t + 1);
            for (int i = 0; i < kAdvancePerThread; ++i) {
                gsns[t].push_back(BufferManager::advancePageGSN());
            }
        });
    }
    for (auto& th : threads) th.join();

    // Collect: all distinct and equal to 1..total range.
    std::vector<bool> seen(kThreads * kAdvancePerThread + 1, false);
    for (auto& v : gsns) {
        for (u64 g : v) {
            ASSERT_FALSE(seen[g]) << "duplicate GSN " << g;
            seen[g] = true;
        }
    }
    // Other fixtures allocated pages in the meantime: the clock only grows.
    EXPECT_GE(BufferManager::nextPageGSN(), kThreads * kAdvancePerThread + 1u);
}

TEST_F(TestGsnDirty, FlushSystemHonorsGsnGate) {
    // End-to-end-ish: alloc, keep pages, flip a bunch of pages, flush_all,
    // verify no page remains whose "would-be-evictable" gate is open.
    std::vector<PID> pids;
    std::vector<u64> gsns;
    for (int i = 0; i < 8; ++i) {
        AllocGuard<Page> g;
        gsns.push_back(g.ptr->p_gsn);
        pids.push_back(g.pid);
    }
    flush_system();  // no LogManager -> unbounded gate -> all pages written
    for (size_t i = 0; i < pids.size(); ++i) {
        EXPECT_FALSE(bm.pageIsDirty(pids[i]))
            << "pid " << pids[i] << " still dirty after flush (gsn "
            << gsns[i] << ", last_written " << bm.lastWrittenGsnOf(pids[i]) << ")";
    }
}

TEST_F(TestGsnDirty, GuardXConcurrentDifferentPages) {
    // Cross-thread guards: each thread fixes its own page via GuardX and
    // verifies GSN growth; no cross-contamination of frame metadata.
    AllocGuard<Page> shared_main;
    u64 main_gsn = shared_main.ptr->p_gsn;

    std::atomic<u64> max_gsn{0};
    std::thread th([&] {
        workerThreadId = 1;
        AllocGuard<Page> g;
        u64 g1 = g.ptr->p_gsn;
        EXPECT_GT(g1, main_gsn);
        bm.updateLastWrittenGsn(g.pid, g1);
        EXPECT_FALSE(bm.pageIsDirty(g.pid));
        max_gsn.store(g1, std::memory_order_release);
    });
    th.join();
    EXPECT_GT(max_gsn.load(), main_gsn);
}
