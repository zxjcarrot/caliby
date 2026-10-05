// Quick sanity tests for log entry header round-tripping and checksums.
#include "recovery/log_entry.hpp"

#include <gtest/gtest.h>

namespace caliby::recovery {

TEST(TestLogEntry, HeaderSizeIs8Bytes) {
    EXPECT_EQ(sizeof(LogEntry), 8u);
    EXPECT_EQ(sizeof(RecoveryInfo), 24u);
    EXPECT_EQ(sizeof(DataEntry), 48u);
}

TEST(TestLogEntry, TypeAndSizePacking) {
    LogEntry e;
    e.type = static_cast<uint8_t>(LogEntry::Type::DATA_ENTRY);
    e.size = sizeof(DataEntry) + 200;
    EXPECT_EQ(e.type, 6);
    EXPECT_EQ(e.size, 248u);
    EXPECT_EQ(e.type_and_size, (e.size << 4) | e.type);
}

TEST(TestLogEntry, ChksumStable) {
    LogEntry e;
    e.type = static_cast<uint8_t>(LogEntry::Type::DATA_ENTRY);
    e.size = 100;
    e.ComputeChksum();
    auto c1 = e.chksum;
    e.ComputeChksum();
    EXPECT_EQ(c1, e.chksum);
}

TEST(TestDataEntry, Layout) {
    DataEntry d;
    d.type = static_cast<uint8_t>(LogEntry::Type::DATA_ENTRY);
    d.size = sizeof(DataEntry) + 32;
    d.rc.w_id = 3;
    d.rc.gsn  = 999;
    d.rc.txn  = 0xABCDEF;
    d.pid     = 12345;
    d.prev_gsn = 998;
    d.ComputeChksum();
    EXPECT_EQ(d.pid, 12345u);
    EXPECT_EQ(d.prev_gsn, 998u);
    EXPECT_EQ(d.rc.gsn, 999u);
    EXPECT_EQ(d.rc.w_id, 3u);
}

}  // namespace caliby::recovery
