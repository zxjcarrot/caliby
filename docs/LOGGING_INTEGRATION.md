# Logging Integration Design (Phased Plan)

Integrates the LeanStore-style WAL into Caliby's vector index / storage stack.
The recovery subsystem skeleton already exists under `src/recovery/` + `include/caliby/recovery/`, but it is **not wired into the runtime** — only tests use it.

---

## Progress checklist

- [x] Phase 1 — Transaction abstraction (`caliby::tx`, no nesting/TxType, sync commit durability)
- [x] Phase 2 — GSN plumbing (`p_gsn` on pages, derived dirty, WAL-gated eviction/flush)
- [ ] Phase 3 — Guard-mediated log emission
- [ ] Phase 4 — Group commit & WAL durability (Phase 1 interim: synchronous fdatasync per commit)
- [ ] Phase 5 — Fuzzy checkpointer (still flushAll stub)
- [ ] Phase 6 — Recovery Analysis + Redo wiring (stubs empty)
- [ ] Phase 7 — Gaps: extents, undo, multi-worker scaling, Python binding surface

Per-phase sections below record what shipped vs. the original draft.

---

## Current state (baseline facts, with pointers)

> **STALE — pre-Phase-1/2 snapshot.** Kept for provenance; the checklist above and the
> phase sections describe the current tree. Baseline facts that changed: `Page::dirty`
> is replaced by first-member `u64 p_gsn` on every page struct; dirty is derived
> (`pageIsDirty` = `p_gsn > lastWrittenGsnOf(pid)`); eviction/flush are WAL-gated;
> `markPageDirty/markPageClean` helpers exist; `flushAll` batches + `markPageWritten`.

- `Page` (calico.hpp:54) carries `bool dirty`; mutated by guard constructors
  (`GuardX` at calico.hpp:1307, `AllocGuard` :1376, OLC upgrade :1321).
- All index mutations happen under `GuardX<T>` / `GuardS<T>`; HNSW neighbor writes via
  `MutableNodeAccessor` (hnsw.hpp:424–484) under `GuardX`.
- `flushAll()` (calico.cpp:1989–2083) writes dirty pages, no GSN gating, no fsync.
- No transaction context, no WAL emit path in production code, no fsync of WAL writes.
- Recovered skeleton: `LogWorker` reservation API (`ReserveDataLog`, `ReserveLogCommitEntry`,
  `ReserveAllocNewPageLogEntry`, ...) in log_worker.hpp; `WalBlockManager` ring;
  `LogManager` singleton; `Checkpointer` currently just calls `flushAll()`
  (recovery/checkpoint.cpp:56–74); `RecoveryManager` has materialize/analyze implemented,
  redo stubs empty (recovery/recovey_manager.cpp:116–125).
- Main-mode plumbing everywhere that matters: none (`rg LogManager` only hits tests).

---

## Phase 1 — Transaction abstraction

**STATUS: DONE.** Shipped vs. original draft:

- Namespace `caliby::tx` (not `caliby::transaction`); thread_local
  `tl_active_txn` + `active_txn()` accessor + `TransactionAutoCommitScope` RAII
  (only controller; `TransactionController` name dropped).
- **No TxType, no nesting** (deprecated per user decision); `EnterTxn` asserts `!IsRunning`.
- `next_txn_id` atomic; `typedefs.hpp` deleted (aliases moved to `recovery/log_entry.hpp`).
- `CommitTransaction()`: TX_COMMIT + serialized GSN vector, then **synchronous** durability
  (LogFlush + fdatasync + AdvanceFlushWatermark) — interim until Phase 4 group commit.
- Start path syncs worker GSN clock to `global_sync_to_this_gsn` (as planned).
- `LocalLogWorker()` = `workerThreadId % workers_.size()` (was worker 0);
  `HasInstance()` added; LogManager ctor calls `SetInstance(this)`.
- Tests: `tests/recovery/test_transaction.cc` (TestTransaction, 8 tests).

Goal: a TLS transaction context that wraps every write operation, exactly like LeanStore's
`transaction::Transaction::active_txn`.

New files: `include/caliby/transaction.hpp`, `src/transaction.cpp`.

```cpp
namespace caliby::transaction {
enum class TxType { SYSTEM, USER };
enum class TxState { STARTED, READY_TO_COMMIT, ABORTED, COMMITTED };

struct Transaction {
    inline static thread_local Transaction active_txn;
    txnid_t txn_id;
    TxType  type; TxState state;
    bool read_only = false;
    // Dependency vector: per-worker highest GSN that previous writers left on pages
    // this txn touched (used later by recovery Analysis + group commit).
    std::unordered_map<wid_t, gsn_t> gsn_vector;
    size_t SerializedVectorSize() const;              // wid array + ts array
    void   SerializeVector(u8* payload) const;        // mirrors transaction.h:63
    recovery::LogWorker& LogWorker() const;           // via LogManager::LocalLogWorker()
    // lifecycle
    void StartTxn(TxType);
    void CommitTransaction();   // ReserveLogCommitEntry(TX_COMMIT) + QueueTransaction()
    void AbortTransaction();    // TX_ABORT w/ vector
    void MarkAsWrite();

    // Write-reentrancy for structural ops that mutate several pages in one guard scope.
    u32   nesting = 0;
    void  BeginNested() { nesting++; }
    bool  EndNested()  { return --nesting == 0; }
};
inline Transaction& active_txn() { return Transaction::active_txn; }
}
```

Design notes (matching LeanStore):

- **Auto-commit for index ops**: for Phase 3 the HNSW/BTree/Collection call sites open a
  `TransactionController` RAII:
  ```cpp
  struct TransactionController {
      TransactionController(TxType t = TxType::USER);
      ~TransactionController();               // auto-commit/abort + LeaveTxn
  };
  ```
  i.e. one txn per `addPoint`/`insert`; explicit multi-op txns can be added later without
  changing the guard API.
- TX_START is only needed for transactional (winner/loser) recovery; for Phase 1 emit
  `TX_START` per txn (cheap, matches LeanStore `transaction.cc:85`).
- `LogWorker()` resolution: fix `LogManager::LocalLogWorker()` (log_manager.cpp:143-146,
  currently always returns worker 0) to honor `LeanStore`-style
  `worker_thread_id` thread_local (calico.hpp:29). Number of LogWorkers must be capped by
  `MAX_NUMBER_OF_WORKERS` (typedefs.hpp:24); assert spawning workers count during
  `LogManager::StartWorkers()`.
- `AdvanceFlushWatermark` / `global_sync_to_this_gsn` plumbing comes in Phase 2; the
  transaction Start path must raise the worker GSN clock to the global checkpoint GSN
  (mirroring transaction_manager.cc:49–56) so worker clocks never lag.
- Keep `CALIBY_WAL_LOGGING` runtime flag; all code guarded so WAL can be disabled.

Acceptance: `Transaction::Start → mutate pages under guards → Commit` runs with
`FLAGS_wal_enable=false` (no-op) and `true` (emits TX_START/TX_COMMIT into a test LogManager),
no page mutation yet touched by WAL. GTests under `tests/recovery/test_transaction.cc`.

---

## Phase 2 — GSN plumbing: page GSN, dirty = (p_gsn > last_written_gsn)

Goal: replace the `bool dirty` dirty-finding path with the LeanStore scheme:

```cpp
PageIsDirty(pid) == (frame[pid].last_written_gsn.load() < ToPtr(pid)->p_gsn)
```

Pieces:

1. **`g_pid` in page payload**: add `u64 p_gsn` to base `Page` (calico.hpp:53-55). All page
   structs inherit it. New pages start at `p_gsn = 1` (LeanStore AllocPage :262).
2. **Frame metadata** (NOT in PageState's u64): parallel array `FrameMeta { atomic<u64> last_written_gsn; wid_t last_writer; }`
   in BufferManager, indexed by PID. Updated at:
   - write-out: `frame[pid].last_written_gsn = page->p_gsn` (see Evict :360)
   - recovery/bootstrap: set to on-disk page p_gsn's value.
3. **Guard changes**: `GuardX` constructor and OLC-upgrade stop setting `dirty = true`
   (calico.hpp:1307, 1321); `AllocGuard` likewise (:1376, 1384). Dirty semantics thereafter
   purely derived from GSN comparison.
4. **Eviction gating (WAL-correctness invariant)**: a dirty page may only be flushed after
   its WAL records are durable:
   ```
   p_gsn <= LogManager::global_min_gsn_flushed   // Evict check: buffer_manager.cc:335
   ```
   Applied in `evict()` (calico.cpp:2109–2195) and `flushAll()` (calico.cpp:1989–2083).
   Pages violating the gate are skipped (not written); write attempt comes after
   `MaybeAutonomousFlush`/checkpointer raise the watermark (Phase 4/5).
5. **`GuardS` read-path GSN sync**: added `DetectGSNDependency()` calls on guard S
   acquisition (Phase 3 will wire the tx dependency vector; here we only add the method).
6. Clean-up: `flushAll()` and per-index flush paths read `PageIsDirty` from GSN comparison;
   `Page::dirty` field removed (grep all users: calico.hpp pages, collection.hpp slots, etc.).

Acceptance: run existing `tests/test_buffer_stress.py`, `test_force_eviction.py`,
`recovery::TestCheckpointer.RespectsDurabilityBarrier`. Behavior identical when
`FLAGS_wal_enable=false` (global_min_gsn_flushed = MAX when WAL disabled so gating is a no-op).

---

## Phase 3 — Guard-mediated log emission (logging only, no redo)

Goal: WAL records generated **through the page guard**, not at mutation call sites —
the invariant enforced in LeanStore `PageGuard::PrepareWalEntry` (page_guard.h:52–93):

> A log entry can only be reserved for a page whose content-generation is under an
> exclusive write guard of the ACTIVE write transaction.

New in Caliby (calico.hpp, alongside GuardX / GuardS):

```cpp
template <class T>
struct GuardX {
    ...
    template <typename WalType>
    WalType& PrepareWalEntry(u64 payload_size);   // calls PrepareDataLogEntry() on worker,
                                                  // sets entry.prev_gsn = page->p_gsn,
                                                  // then AdvanceGSN()
    void        DetectGSNDependency();
    void        AdvanceGSN();   // page->p_gsn = logger.GetCurrentGSN()+1;
                                // frame last_writer = worker_thread_id
};
struct DeferLog { ... };  // payload assembled under X-guard, submitted after unfixX
                          // (LeanStore page_guard.h:108) — needed for HNSW's
                          // optimistic-link retry loops.
```

Reservation ordering rule (page_guard.h:59–69, port verbatim):
```
assert(mode == GuardMode::EXCLUSIVE)
assert(active_txn.IsRunning() and not ReadOnly())
prev_gsn = ptr->p_gsn
AdvanceGSN()
entry = logger.ReserveDataLog(...); entry.prev_gsn = prev_gsn
// caller then mutates page payload, then calls SubmitActiveLogEntry()
```

**Log record set (BTree first, then HNSW) — `include/caliby/recovery/log_entry.hpp` gains:**

```cpp
enum class WalType : u8 {  // mirrors storage/btree/wal.h:12
    INSERT = 0, AFTER_IMAGE = 1, DELTA_IMAGE = 2, REMOVE = 3,
    LOGICAL_SPLIT = 4, NEW_PAGE = 6, FRESH_PAGE = 7, NEW_ROOT = 8, INSERT_SEP = 9,
};
struct WALEntry : recovery::DataEntry { WalType wal_type : 4; };
// + the 8 concrete record structs (WALInsert, WALRemove, WALInsertSep,
//   WALLogicalSplit, WALAfterImage, WALDeltaImage, WALNewPage, WALFreshPage, WALNewRoot)
```
Matched generator helpers in `include/caliby/btree/wal.hpp` (GenerateWAL / GenerateWALNewRoot /
GenerateWALNewPage / GenerateWALFreshPage — port of `storage/btree/wal.h:81–117`).

**BTree sites** (calico.cpp:2792–2982): `insert` (AFTER_IMAGE / INSERT / INSERT_SEP),
`remove` (REMOVE), `updateInPlace` (AFTER_IMAGE / DELTA_IMAGE), `updateOutOfPlace`,
`SplitNode` (NEW_PAGE child + LOGICAL_SPLIT / INSERT_SEP), root changes
(`NEW_ROOT` on the MetaDataPage + ALLOC_NEW_PAGE).

**HNSW sites** (hnsw.cpp `addPoint_internal` 704–1138):
- page first-use (`on-demand alloc` 709–756) → `AllocNewPageEntry` (pid=0, new_page_id) +
  `ReserveAllocNewPageLogEntry`, then `WALFreshPage`-style full-page first image for the
  HNSWPage layout (use NEW_PAGE variant, since HNSW pages have a different layout than BTree).
- link updates under `GuardX` (`setNeighbors`/`addNeighbor` hnsw.hpp:424–484) →
  wrapper `WALAfterImage`-style record keyed by `(node_id, level)`.
- metadata updates (entry point / max_level / node_count bump, `HNSWMetadataPage`) →
  `NEW_ROOT`-equivalent `LOG_META` record or full metadata page image (NEW_PAGE).
- Alloc-only records are excluded from GSN tracking (they are order-independent and
  idempotent, like `AllocNewPageEntry` — log_entry.h:120–128).

**Collection sites** (collection.cpp `write_document` ~2100s): page-image records on
DocumentPage/slot modifications; id-index BTree routes through the BTree sites.

Sub-cases:

- **WILO/concurrent append-only flush inside EnsureEnoughSpace** can trigger a flush while
  an X guard is held → `GuardX` holds `active_log` cursor and calls `SubmmitActiveLogEntry()`
  only after mutation (Use DeferLog for all payloads > ~1KB).
- All emission behind `if (FLAGS_wal_enable && !txn.read_only)`.

Acceptance: GTest printing/inspecting raw WAL for a scripted insert sequence (`RoundTripLogEntry`-
style, extending `tests/recovery/test_log_entry.cc`); python smoke `pytest tests/test_recovery.py`
(unaffected semantics since recovery/redo not yet wired).

---

## Phase 4 — Group commit & WAL durability (fsync + watermarks)

Port `GroupCommitExecutor` (`./LeanStore/src/recovery/group_commit.cc`):

- one per `worker_count / caliby_num_commit_group`, io_uring ring, SPSC
  `precommitted_queue` (transaction_manager.cc:174–180 `QueueTransaction` →
  `TriggerGroupCommit` on `SubmitActiveLogEntry()==true` or `FLAGS_wal_force_log_flush`).
- 3 phases per round, `CollectConsistentState()` + `WorkerConsistentState.SyncClone`
  (WorkerConsistentState needs a HybridLatch — log_worker.hpp:22–36 currently has none).
- round tail: `fdatasync(wal_fd_)` **once per round, not per entry** (group_commit.cc:177),
  then `AdvanceFlushWatermark()` raises `global_min_gsn_flushed` (log_manager.cpp:153–168),
  `DurableCommit()` wakes up txn waiters.
- WAL write path: `LogWorker::FlushRange` (log_worker.cpp:150–188) uses io_uring
  (`LogBackend` with per-worker ring, port of `log_backend.cc`) instead of current
  pwrite_at; WAL file opened with O_DIRECT.
- Eviction gate from Phase 2 becomes meaningful: pages may now actually advance to disk
  as group commit raises the frontier.

Acceptance: multi-threaded insert + crash-without-flush test asserting that all pages with
`p_gsn <= global_min_gsn_flushed` were written out.

---

## Phase 5 — Fuzzy checkpointer (sharded, coverage-based)

Replace `Checkpointer::TryCheckpoint` current "flushAll()" stub with LeanStore's
coverage-correct version (checkpoint.cc:87–117 + buffer_manager.cc:428–479):

- N shards (≥ worker count) round-robin; for each shard scan resident pages:
  flush pages where `(p_gsn <= target_gsn) || (last_written_gsn <= target_gsn)`; shard
  watermark = min over shard's pages of the **on-disk coverage** (`shard_min_coverage`) —
  never `target_gsn` itself. (Comment buffer_manager.cc:419–427 explains the stale-image
  bug if correctness rule violated.)
- whole-DB checkpoint watermark = min over shard watermarks → `MarkDoneThroughGsn`.
- `Bootstrap()` after recovery (checkpoint.cc:119–137): per-shard watermarks from
  `ScanRecoveryCoverage()`; prune CLOSED blocks.

Acceptance: WAL ring wrap-around test: 100× the WAL capacity worth of writes; back-pressure
of `OpenNext` verified with `FLAGS_wal_enable_checkpoint=false` (must throw — checkpoint.cc:49).

---

## Phase 6 — Recovery: Analysis + Redo wired into startup

Port build blocks in order (LeanStore `leanstore.cc:58–123` `.cpp` + recovery_manager.cc):

1. Master record: `ReadMasterRecord` on startup (log_manager.cpp:190–199); write on fresh
   session (`WriteMasterRecord` :170). Extend fields:
   `num_workers, wal_size, wal_block_size, signature, max_pid, next_gsn` to also carry
   per-index alloc snapshots (`alloc_count_snapshot` currently restored at
   calico.cpp:3051–3066 — move into MASTER_RECORD).
2. `initialize_system()` (calico.cpp:3026–3068) gains:
   - if META exists: `RecoveryPreparation` → `MaterializeLogs` (parallel, #recovery threads)
     → `ApplyRecoveredBlocks` → `Analysis` (winners/losers via `global_durable_gsn_[wid]`
     vs commit GSN vectors) → `PrepareRedoEnv` (metadata page first; `RedoFreeExtents`) →
     parallel `Redo` (`PerPageRedo`: merge per-pid logs in GSN order via `LazyGenerator`,
     skip losers, stop at `p_gsn`).
   - `has_recovered_[pid]` state + `HasRecovered(pid)` check in `handleFault` (only used
     by recovery threads — main flow unchanged).
3. **RedoLog per record type** implemented for every WalType of Phase 3 — mirroring
   recovery_manager.cc:275–354. HNSW-specific: apply node-image (NEW_PAGE) then
   neighbor-link records (AFTER_IMAGE for a `(node_id, level)` pair) — order enforced by
   GSN.
4. `CALIBY_WAL_RECOVERY` flag gates the whole phase; when disabled catalog-only recovery
   (today's `recovered_from_disk_` path in hnsw.cpp:156–163, ivfpq.cpp:625–642) stays as-is.

Acceptance (final gate for the whole project): the existing
`tests/test_crash_recovery_persistence.py` switched to abrupt-kill mode keeps passing with
`FLAGS_wal_enable_recovery=true`, plus GTests `test_recovery.cc` extended to whose phases
(materialize → analysis → redo) end-to-end.

---

## Phase 7 — Remaining gaps (tracked, not scheduled here)

- `FreeExtent/ReuseExtent` for pid reuse/portions (CONTRIBUTING_TODO.md:130–148) —
  needed before any delete/reuse exists in HNSW (currently append-only).
- `Undo()` (CLR-based) — only required if isolation > READ_UNCOMMITTED is introduced.
- Multi-worker scaling: `LocalLogWorker()` TLS correctness, `MAX_NUMBER_OF_WORKERS=128`
  cap, WAL region sizing against worker counts, GSN vector size = workers.
- Python binding surface: `caliby.set_wal_config()`; expose `txn` context-manager.

---

## Ordering / dependency graph

```
P1 transactions ─┬─→ P3 WAL emission (needs log entry types + guards from P2)
P2 GSN plumbing ─┘
P3 ─→ P4 group commit ─→ P5 checkpointer ─→ P6 recovery wiring ─→ done
                                     (P7 items may interleave)
```

Reference LeanStore mapping table:

| Caliby target          | LeanStore source                                  |
|------------------------|-------------------------------------------------------|
| transaction.hpp/cpp    | src/include/transaction/transaction.h, transaction_manager.cc |
| FrameMeta/dirty        | src/buffer/buffer_manager.cc:225, 262                 |
| GuardX::PrepareWalEntry| src/sync/page_guard/page_guard.h:52                   |
| WAL record structs     | src/include/storage/btree/wal.h                       |
| BTree WAL sites        | src/storage/btree/tree.cc (31–116, 145, 215, 316)     |
| DetectGSNDependency    | src/sync/page_guard/page_guard.cc:71–111              |
| Group commit           | src/recovery/group_commit.cc                          |
| Checkpointer           | src/recovery/checkpoint.cc:87–137                     |
| Recovery algorithm     | src/recovery/recovery_manager.cc, src/bin/leanstore.cc:58–123 |
