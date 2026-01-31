/**
 * @file text_index.hpp
 * @brief BM25 Text Index for Caliby Collection System
 * 
 * Implements an inverted index with BM25 scoring for full-text search.
 * Features:
 * - Term dictionary (persistent B-tree)
 * - Posting lists stored in BTree leaf pages with chaining for overflow
 * - Skip pointers for fast iteration
 * - Document length stored inline with postings for O(1) BM25 scoring
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>
#include <optional>

#include "calico.hpp"

namespace caliby {

//=============================================================================
// Constants
//=============================================================================

constexpr uint64_t TEXT_INDEX_MAGIC = 0xCA11B77E71D00000ULL;  // "CALIBYTEXT"
constexpr uint32_t TEXT_INDEX_VERSION = 2;  // Version 2 with persistent BTree
constexpr size_t SKIP_INTERVAL = 128;          // Skip pointer every N entries
constexpr size_t MAX_TERM_LENGTH = 256;        // Maximum term length

// Maximum posting list size that can fit in a single BTree leaf payload
// BTree leaf has ~3.5KB usable space, we reserve some for key
constexpr size_t MAX_INLINE_POSTING_BYTES = 3000;

//=============================================================================
// Text Analyzer
//=============================================================================

/**
 * Analyzer type for text tokenization.
 */
enum class AnalyzerType : uint8_t {
    STANDARD = 0,    // Lowercase, split on whitespace/punctuation
    WHITESPACE = 1,  // Split on whitespace only
    NONE = 2,        // No tokenization (exact match)
};

/**
 * Simple text analyzer.
 */
class TextAnalyzer {
public:
    TextAnalyzer(AnalyzerType type = AnalyzerType::STANDARD,
                 const std::string& language = "english");
    
    /**
     * Tokenize text into terms.
     */
    std::vector<std::string> tokenize(const std::string& text) const;
    
    /**
     * Analyze text: tokenize + stem + lowercase.
     */
    std::vector<std::string> analyze(const std::string& text) const;
    
private:
    AnalyzerType type_;
    std::string language_;
    
    // Simple lowercase and punctuation handling
    std::string normalize_term(const std::string& term) const;
};

//=============================================================================
// Persistent Posting List Structures
//=============================================================================

/**
 * A compact posting entry stored in BTree payload.
 * 10 bytes per posting (no positions for simplicity).
 */
struct CompactPosting {
    uint64_t doc_id;      // 8 bytes - Document ID
    uint16_t term_freq;   // 2 bytes - Term frequency in document
} __attribute__((packed));

static_assert(sizeof(CompactPosting) == 10, "CompactPosting must be 10 bytes");

/**
 * Header for posting list stored in BTree payload.
 * Format: [PostingListHeader][CompactPosting...][optional overflow chain pointer]
 */
struct PostingListHeader {
    uint32_t doc_freq;           // 4 bytes - Number of documents containing term  
    uint32_t num_postings;       // 4 bytes - Number of postings in this chunk
    uint32_t overflow_slot_id;   // 4 bytes - BTree slot ID for overflow (0 = none)
    uint8_t  is_overflow;        // 1 byte  - 1 if this is an overflow chunk
    uint8_t  reserved[3];        // 3 bytes - Padding
} __attribute__((packed));

static_assert(sizeof(PostingListHeader) == 16, "PostingListHeader must be 16 bytes");

/**
 * Calculate max postings that can fit in a given payload size.
 */
inline constexpr size_t maxPostingsInPayload(size_t payloadSize) {
    if (payloadSize <= sizeof(PostingListHeader)) return 0;
    return (payloadSize - sizeof(PostingListHeader)) / sizeof(CompactPosting);
}

/**
 * In-memory posting list for building/searching.
 */
class PostingList {
public:
    PostingList() = default;
    
    /**
     * Add a posting to the list.
     */
    void add(uint64_t doc_id, uint16_t term_freq);
    
    /**
     * Get document frequency (number of documents containing term).
     */
    size_t doc_freq() const { return postings_.size(); }
    
    /**
     * Get all postings.
     */
    const std::vector<CompactPosting>& postings() const { return postings_; }
    
    /**
     * Serialize to bytes for BTree payload (single chunk).
     * Returns serialized data and whether overflow is needed.
     */
    std::vector<uint8_t> serialize(size_t maxBytes, size_t* numSerialized = nullptr) const;
    
    /**
     * Serialize remaining postings starting from offset (for overflow chunks).
     */
    std::vector<uint8_t> serializeFrom(size_t startOffset, size_t maxBytes, 
                                        size_t* numSerialized = nullptr) const;
    
    /**
     * Deserialize from BTree payload bytes (single chunk).
     * Returns the PostingListHeader for chaining info.
     */
    static PostingListHeader deserialize(const uint8_t* data, size_t len, 
                                          std::vector<CompactPosting>& postings);
    
    /**
     * Clear the posting list.
     */
    void clear() { postings_.clear(); }
    
    /**
     * Merge another posting list into this one (for combining chunks).
     */
    void merge(const std::vector<CompactPosting>& other);
    
private:
    std::vector<CompactPosting> postings_;
};

//=============================================================================
// Text Index Metadata Page
//=============================================================================

/**
 * Text index metadata stored in collection metadata.
 * This is now more compact as the BTree handles its own persistence.
 */
struct TextIndexMetadata {
    uint64_t magic;                     // TEXT_INDEX_MAGIC
    uint32_t version;                   // TEXT_INDEX_VERSION
    uint32_t btree_slot_id;             // BTree slot ID for term dictionary
    
    uint64_t vocab_size;                // Number of unique terms
    uint64_t doc_count;                 // Number of indexed documents
    uint64_t total_doc_length;          // Total of all document lengths
    
    uint8_t analyzer_type;              // AnalyzerType enum
    uint8_t reserved1[3];
    
    // BM25 parameters
    float k1;                           // BM25 k1 (default 1.2)
    float b;                            // BM25 b (default 0.75)
    
    uint8_t reserved2[20];              // Reserved for future use
    
    void initialize() {
        magic = TEXT_INDEX_MAGIC;
        version = TEXT_INDEX_VERSION;
        btree_slot_id = 0;
        vocab_size = 0;
        doc_count = 0;
        total_doc_length = 0;
        analyzer_type = static_cast<uint8_t>(AnalyzerType::STANDARD);
        std::memset(reserved1, 0, sizeof(reserved1));
        k1 = 1.2f;
        b = 0.75f;
        std::memset(reserved2, 0, sizeof(reserved2));
    }
    
    bool is_valid() const {
        return magic == TEXT_INDEX_MAGIC && version == TEXT_INDEX_VERSION;
    }
    
    float avg_doc_len() const {
        return doc_count > 0 ? static_cast<float>(total_doc_length) / doc_count : 0.0f;
    }
} __attribute__((packed));

static_assert(sizeof(TextIndexMetadata) == 72, "TextIndexMetadata size check");

//=============================================================================
// BM25 Scorer
//=============================================================================

/**
 * BM25 scoring function.
 */
class BM25Scorer {
public:
    BM25Scorer(float k1 = 1.2f, float b = 0.75f);
    
    /**
     * Set corpus statistics.
     */
    void set_corpus_stats(uint64_t doc_count, float avg_doc_len);
    
    /**
     * Calculate BM25 score for a term in a document.
     * @param term_freq Term frequency in document
     * @param doc_len Document length (number of terms) - passed as parameter now
     * @param doc_freq Document frequency of term
     * @return BM25 score component for this term
     */
    float score(uint16_t term_freq, uint32_t doc_len, uint32_t doc_freq) const;
    
private:
    float k1_;
    float b_;
    uint64_t doc_count_ = 0;
    float avg_doc_len_ = 0.0f;
};

//=============================================================================
// Text Index Class
//=============================================================================

/**
 * BM25-based text search index with persistent BTree storage.
 */
class TextIndex {
public:
    /**
     * Create a new text index.
     * @param collection_id Parent collection ID  
     * @param index_id This index's ID
     * @param analyzer_type Tokenization method
     * @param language Language for stemming
     * @param k1 BM25 k1 parameter
     * @param b BM25 b parameter
     */
    TextIndex(uint32_t collection_id,
              uint32_t index_id,
              AnalyzerType analyzer_type = AnalyzerType::STANDARD,
              const std::string& language = "english",
              float k1 = 1.2f,
              float b = 0.75f);
    
    /**
     * Open an existing text index from BTree slot ID.
     */
    static std::unique_ptr<TextIndex> open(uint32_t index_id, 
                                            uint32_t btree_slot_id,
                                            uint64_t vocab_size,
                                            uint64_t doc_count,
                                            uint64_t total_doc_length,
                                            float k1 = 1.2f,
                                            float b = 0.75f);
    
    ~TextIndex();
    
    // Non-copyable
    TextIndex(const TextIndex&) = delete;
    TextIndex& operator=(const TextIndex&) = delete;
    
    //-------------------------------------------------------------------------
    // Indexing Operations
    //-------------------------------------------------------------------------
    
    /**
     * Index a document's text content.
     * @param doc_id Document ID
     * @param content Text content to index
     * @param update_doc_length Callback to update doc length in Collection (optional)
     */
    void index_document(uint64_t doc_id, const std::string& content,
                        const std::function<void(uint64_t, uint32_t)>* update_doc_length = nullptr);
    
    /**
     * Remove a document from the index.
     * @param doc_id Document ID to remove
     */
    void remove_document(uint64_t doc_id);
    
    /**
     * Batch index multiple documents.
     * @param doc_ids Document IDs
     * @param contents Text content for each document
     * @param update_doc_length Callback to update doc length in Collection (optional)
     */
    void index_batch(const std::vector<uint64_t>& doc_ids,
                     const std::vector<std::string>& contents,
                     const std::function<void(uint64_t, uint32_t)>* update_doc_length = nullptr);
    
    //-------------------------------------------------------------------------
    // Search Operations
    //-------------------------------------------------------------------------
    
    /**
     * Search for documents matching query text.
     * @param query Query text
     * @param k Number of results
     * @param doc_filter Optional bitmap of valid doc IDs (nullptr = all)
     * @param doc_lengths Map of doc_id -> doc_length for BM25 scoring
     * @return Vector of (doc_id, score) pairs, sorted by score descending
     */
    std::vector<std::pair<uint64_t, float>> search(
        const std::string& query,
        size_t k,
        const std::vector<uint64_t>* doc_filter = nullptr,
        const std::function<uint32_t(uint64_t)>* get_doc_length = nullptr) const;
    
    //-------------------------------------------------------------------------
    // Statistics & Persistence
    //-------------------------------------------------------------------------
    
    /**
     * Get vocabulary size (number of unique terms).
     */
    uint64_t vocab_size() const { return vocab_size_.load(); }
    
    /**
     * Get number of indexed documents.
     */
    uint64_t doc_count() const { return doc_count_.load(); }
    
    /**
     * Get total document length for avg computation.
     */
    uint64_t total_doc_length() const { return total_doc_length_.load(); }
    
    /**
     * Get average document length.
     */
    float avg_doc_len() const {
        uint64_t count = doc_count_.load();
        return count > 0 ? static_cast<float>(total_doc_length_.load()) / count : 0.0f;
    }
    
    /**
     * Get BTree slot ID for persistence.
     */
    uint32_t btree_slot_id() const;
    
    /**
     * Flush changes to disk.
     */
    void flush();
    
    /**
     * Record a document's length for statistics.
     * Called by Collection when indexing documents.
     */
    void record_doc_length(uint64_t doc_id, uint32_t length);
    
private:
    uint32_t collection_id_;
    uint32_t index_id_;
    TextAnalyzer analyzer_;
    BM25Scorer scorer_;
    
    // Persistent term dictionary BTree
    // Key: term string (variable length)
    // Value: PostingListHeader + CompactPosting[]
    std::unique_ptr<BTree> term_btree_;
    
    // Thread safety
    mutable std::shared_mutex mutex_;
    
    // Statistics (persisted via Collection metadata)
    std::atomic<uint64_t> vocab_size_{0};
    std::atomic<uint64_t> doc_count_{0};
    std::atomic<uint64_t> total_doc_length_{0};
    
    // BM25 parameters
    float k1_;
    float b_;
    
    // Internal methods
    void insert_or_update_posting(const std::string& term, uint64_t doc_id, uint16_t term_freq);
    PostingList load_posting_list(const std::string& term) const;
    void save_posting_list(const std::string& term, const PostingList& list);
};

} // namespace caliby
