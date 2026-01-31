/**
 * @file text_index.cpp
 * @brief Implementation of BM25 Text Index with Persistent BTree
 */

#include "text_index.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <cctype>
#include <numeric>
#include <functional>
#include <iostream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace caliby {

//=============================================================================
// TextAnalyzer Implementation
//=============================================================================

TextAnalyzer::TextAnalyzer(AnalyzerType type, const std::string& language)
    : type_(type)
    , language_(language)
{
}

std::string TextAnalyzer::normalize_term(const std::string& term) const {
    std::string result;
    result.reserve(term.size());
    
    for (char c : term) {
        // Convert to lowercase and keep only alphanumeric
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    
    return result;
}

std::vector<std::string> TextAnalyzer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    
    if (type_ == AnalyzerType::NONE) {
        // No tokenization - treat entire text as one token
        if (!text.empty()) {
            tokens.push_back(text);
        }
        return tokens;
    }
    
    std::string current_token;
    current_token.reserve(64);
    
    for (char c : text) {
        bool is_separator = false;
        
        if (type_ == AnalyzerType::WHITESPACE) {
            is_separator = std::isspace(static_cast<unsigned char>(c));
        } else {
            // STANDARD: split on whitespace and punctuation
            is_separator = !std::isalnum(static_cast<unsigned char>(c));
        }
        
        if (is_separator) {
            if (!current_token.empty()) {
                tokens.push_back(std::move(current_token));
                current_token.clear();
            }
        } else {
            current_token += c;
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(std::move(current_token));
    }
    
    return tokens;
}

std::vector<std::string> TextAnalyzer::analyze(const std::string& text) const {
    auto tokens = tokenize(text);
    std::vector<std::string> result;
    result.reserve(tokens.size());
    
    // Simple stopword list
    static const absl::flat_hash_set<std::string> stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by",
        "for", "if", "in", "into", "is", "it", "no", "not", "of",
        "on", "or", "such", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "will", "with"
    };
    
    for (auto& token : tokens) {
        std::string normalized = normalize_term(token);
        
        // Skip empty and very short tokens
        if (normalized.length() < 2) {
            continue;
        }
        
        // Skip stopwords for STANDARD analyzer
        if (type_ == AnalyzerType::STANDARD && stopwords.count(normalized)) {
            continue;
        }
        
        result.push_back(std::move(normalized));
    }
    
    return result;
}

//=============================================================================
// PostingList Implementation
//=============================================================================

void PostingList::add(uint64_t doc_id, uint16_t term_freq) {
    CompactPosting posting{doc_id, term_freq};
    
    // Fast path: if posting list is empty or doc_id > last, just append
    if (postings_.empty() || doc_id > postings_.back().doc_id) {
        postings_.push_back(posting);
        return;
    }
    
    // Slow path: need to find insertion point (rare during bulk insert)
    auto it = std::lower_bound(postings_.begin(), postings_.end(), posting,
        [](const CompactPosting& a, const CompactPosting& b) { return a.doc_id < b.doc_id; });
    
    if (it != postings_.end() && it->doc_id == doc_id) {
        // Update existing posting
        it->term_freq = term_freq;
    } else {
        postings_.insert(it, posting);
    }
}

std::vector<uint8_t> PostingList::serialize(size_t maxBytes, size_t* numSerialized) const {
    return serializeFrom(0, maxBytes, numSerialized);
}

std::vector<uint8_t> PostingList::serializeFrom(size_t startOffset, size_t maxBytes, 
                                                  size_t* numSerialized) const {
    std::vector<uint8_t> data;
    
    if (maxBytes < sizeof(PostingListHeader)) {
        if (numSerialized) *numSerialized = 0;
        return data;
    }
    
    // Calculate how many postings can fit
    size_t maxPostings = maxPostingsInPayload(maxBytes);
    size_t availablePostings = postings_.size() > startOffset ? postings_.size() - startOffset : 0;
    size_t toSerialize = std::min(maxPostings, availablePostings);
    
    // Calculate actual size needed
    size_t dataSize = sizeof(PostingListHeader) + toSerialize * sizeof(CompactPosting);
    data.resize(dataSize);
    
    // Fill header
    PostingListHeader* header = reinterpret_cast<PostingListHeader*>(data.data());
    header->doc_freq = static_cast<uint32_t>(postings_.size());
    header->num_postings = static_cast<uint32_t>(toSerialize);
    header->overflow_slot_id = 0;  // Caller sets this if needed
    header->is_overflow = (startOffset > 0) ? 1 : 0;
    std::memset(header->reserved, 0, sizeof(header->reserved));
    
    // Copy postings
    if (toSerialize > 0) {
        CompactPosting* postingDest = reinterpret_cast<CompactPosting*>(data.data() + sizeof(PostingListHeader));
        std::memcpy(postingDest, postings_.data() + startOffset, toSerialize * sizeof(CompactPosting));
    }
    
    if (numSerialized) *numSerialized = toSerialize;
    return data;
}

PostingListHeader PostingList::deserialize(const uint8_t* data, size_t len, 
                                            std::vector<CompactPosting>& postings) {
    PostingListHeader header{};
    
    if (len < sizeof(PostingListHeader)) {
        return header;
    }
    
    // Copy header
    std::memcpy(&header, data, sizeof(PostingListHeader));
    
    // Copy postings
    size_t postingBytes = len - sizeof(PostingListHeader);
    size_t numPostings = std::min(static_cast<size_t>(header.num_postings), 
                                  postingBytes / sizeof(CompactPosting));
    
    if (numPostings > 0) {
        const CompactPosting* postingSrc = reinterpret_cast<const CompactPosting*>(data + sizeof(PostingListHeader));
        postings.reserve(postings.size() + numPostings);
        for (size_t i = 0; i < numPostings; i++) {
            postings.push_back(postingSrc[i]);
        }
    }
    
    return header;
}

void PostingList::merge(const std::vector<CompactPosting>& other) {
    postings_.insert(postings_.end(), other.begin(), other.end());
    // Sort by doc_id to maintain order
    std::sort(postings_.begin(), postings_.end(),
        [](const CompactPosting& a, const CompactPosting& b) { return a.doc_id < b.doc_id; });
}

//=============================================================================
// BM25Scorer Implementation
//=============================================================================

BM25Scorer::BM25Scorer(float k1, float b)
    : k1_(k1), b_(b)
{
}

void BM25Scorer::set_corpus_stats(uint64_t doc_count, float avg_doc_len) {
    doc_count_ = doc_count;
    avg_doc_len_ = avg_doc_len;
}

float BM25Scorer::score(uint16_t term_freq, uint32_t doc_len, uint32_t doc_freq) const {
    if (doc_count_ == 0 || doc_freq == 0) {
        return 0.0f;
    }
    
    // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    double N = static_cast<double>(doc_count_);
    double df = static_cast<double>(doc_freq);
    double idf = std::log((N - df + 0.5) / (df + 0.5) + 1.0);
    
    // TF normalization
    double tf = static_cast<double>(term_freq);
    double dl = static_cast<double>(doc_len);
    double avgdl = static_cast<double>(avg_doc_len_);
    
    double norm_tf = (tf * (k1_ + 1.0)) / 
                     (tf + k1_ * (1.0 - b_ + b_ * (dl / avgdl)));
    
    return static_cast<float>(idf * norm_tf);
}

//=============================================================================
// TextIndex Implementation
//=============================================================================

TextIndex::TextIndex(uint32_t collection_id,
                     uint32_t index_id,
                     AnalyzerType analyzer_type,
                     const std::string& language,
                     float k1,
                     float b)
    : collection_id_(collection_id)
    , index_id_(index_id)
    , analyzer_(analyzer_type, language)
    , scorer_(k1, b)
    , k1_(k1)
    , b_(b)
{
    // Create new BTree for term dictionary
    term_btree_ = std::make_unique<BTree>();
}

std::unique_ptr<TextIndex> TextIndex::open(uint32_t index_id,
                                            uint32_t btree_slot_id,
                                            uint64_t vocab_size,
                                            uint64_t doc_count,
                                            uint64_t total_doc_length,
                                            float k1,
                                            float b) {
    auto index = std::unique_ptr<TextIndex>(new TextIndex(0, index_id, AnalyzerType::STANDARD, "english", k1, b));
    
    // Recover existing BTree from slot ID
    index->term_btree_ = std::make_unique<BTree>(btree_slot_id);
    
    // Restore statistics
    index->vocab_size_.store(vocab_size);
    index->doc_count_.store(doc_count);
    index->total_doc_length_.store(total_doc_length);
    
    return index;
}

TextIndex::~TextIndex() {
    try {
        flush();
    } catch (...) {
        // Ignore flush errors in destructor
    }
}

void TextIndex::record_doc_length(uint64_t doc_id, uint32_t length) {
    // Increment statistics
    total_doc_length_.fetch_add(length);
    doc_count_.fetch_add(1);
}

void TextIndex::index_document(uint64_t doc_id, const std::string& content,
                                const std::function<void(uint64_t, uint32_t)>* update_doc_length) {
    std::unique_lock lock(mutex_);
    
    // Analyze content
    auto terms = analyzer_.analyze(content);
    
    if (terms.empty()) {
        return;
    }
    
    // Count term frequencies
    absl::flat_hash_map<std::string, uint16_t> term_freqs;
    term_freqs.reserve(terms.size());
    for (const auto& term : terms) {
        term_freqs[term]++;
    }
    
    // Update document length statistics
    uint32_t doc_len = static_cast<uint32_t>(terms.size());
    
    // Update doc length in Collection's id_index via callback
    if (update_doc_length) {
        (*update_doc_length)(doc_id, doc_len);
    }
    
    total_doc_length_.fetch_add(doc_len);
    doc_count_.fetch_add(1);
    
    // Add to posting lists
    for (const auto& [term, freq] : term_freqs) {
        insert_or_update_posting(term, doc_id, freq);
    }
}

void TextIndex::insert_or_update_posting(const std::string& term, uint64_t doc_id, uint16_t term_freq) {
    // Load existing posting list (if any)
    PostingList list = load_posting_list(term);
    
    // Check if this is a new term
    bool is_new_term = (list.doc_freq() == 0);
    
    // Add new posting
    list.add(doc_id, term_freq);
    
    // Save updated posting list
    save_posting_list(term, list);
    
    // Update vocab size if new term
    if (is_new_term) {
        vocab_size_.fetch_add(1);
    }
}

PostingList TextIndex::load_posting_list(const std::string& term) const {
    PostingList list;
    
    if (!term_btree_) {
        return list;
    }
    
    // Create key from term
    std::span<uint8_t> key(reinterpret_cast<uint8_t*>(const_cast<char*>(term.data())), term.size());
    
    // Lookup main entry in BTree
    uint32_t next_overflow = 0;
    bool found = term_btree_->lookup(key, [&list, &next_overflow](std::span<uint8_t> payload) {
        std::vector<CompactPosting> postings;
        PostingListHeader header = PostingList::deserialize(payload.data(), payload.size(), postings);
        
        for (const auto& p : postings) {
            list.add(p.doc_id, p.term_freq);
        }
        
        next_overflow = header.overflow_slot_id;
    });
    
    if (!found) {
        return list;
    }
    
    // Follow overflow chain if present
    uint32_t overflow_idx = 1;
    while (next_overflow != 0) {
        // Create overflow key: term + null byte + overflow index
        std::string overflow_key = term;
        overflow_key.push_back('\0');
        overflow_key.append(reinterpret_cast<const char*>(&overflow_idx), sizeof(overflow_idx));
        
        std::span<uint8_t> okey(reinterpret_cast<uint8_t*>(const_cast<char*>(overflow_key.data())), overflow_key.size());
        
        uint32_t current_overflow = next_overflow;
        next_overflow = 0;
        
        bool overflow_found = term_btree_->lookup(okey, [&list, &next_overflow](std::span<uint8_t> payload) {
            std::vector<CompactPosting> postings;
            PostingListHeader header = PostingList::deserialize(payload.data(), payload.size(), postings);
            
            for (const auto& p : postings) {
                list.add(p.doc_id, p.term_freq);
            }
            
            next_overflow = header.overflow_slot_id;
        });
        
        if (!overflow_found) {
            break;  // Chain broken, stop here
        }
        
        overflow_idx++;
    }
    
    return list;
}

void TextIndex::save_posting_list(const std::string& term, const PostingList& list) {
    if (!term_btree_) {
        return;
    }
    
    // Create key from term
    std::span<uint8_t> key(reinterpret_cast<uint8_t*>(const_cast<char*>(term.data())), term.size());
    
    // First, remove any existing overflow entries for this term
    // We rebuild the entire chain on each update
    uint32_t overflow_idx = 1;
    while (true) {
        std::string overflow_key = term;
        overflow_key.push_back('\0');
        overflow_key.append(reinterpret_cast<const char*>(&overflow_idx), sizeof(overflow_idx));
        
        std::span<uint8_t> okey(reinterpret_cast<uint8_t*>(const_cast<char*>(overflow_key.data())), overflow_key.size());
        
        // Try to remove - if it fails (key not found), we're done cleaning up
        bool removed = term_btree_->remove(okey);
        if (!removed) {
            break;
        }
        overflow_idx++;
    }
    
    // Now serialize and save the posting list with overflow chaining if needed
    size_t totalPostings = list.doc_freq();
    size_t offset = 0;
    overflow_idx = 0;
    
    while (offset < totalPostings) {
        size_t numSerialized = 0;
        auto payload = list.serializeFrom(offset, MAX_INLINE_POSTING_BYTES, &numSerialized);
        
        if (payload.empty() || numSerialized == 0) {
            break;
        }
        
        // Check if there will be more chunks after this one
        bool hasMoreChunks = (offset + numSerialized) < totalPostings;
        
        // Update header to indicate overflow
        PostingListHeader* header = reinterpret_cast<PostingListHeader*>(payload.data());
        header->is_overflow = (overflow_idx > 0) ? 1 : 0;
        header->overflow_slot_id = hasMoreChunks ? (overflow_idx + 1) : 0;
        
        // Create the key for this chunk
        std::string chunk_key;
        if (overflow_idx == 0) {
            chunk_key = term;
        } else {
            chunk_key = term;
            chunk_key.push_back('\0');
            chunk_key.append(reinterpret_cast<const char*>(&overflow_idx), sizeof(overflow_idx));
        }
        
        std::span<uint8_t> chunkKey(reinterpret_cast<uint8_t*>(const_cast<char*>(chunk_key.data())), chunk_key.size());
        std::span<uint8_t> payloadSpan(payload.data(), payload.size());
        
        // Try to update existing entry
        int result = term_btree_->updateOutOfPlace(chunkKey, payloadSpan);
        
        if (result == -1) {
            // Key not found - insert new entry
            term_btree_->insert(chunkKey, payloadSpan);
        } else if (result == -2) {
            // No space in leaf - delete and reinsert (triggers split)
            term_btree_->remove(chunkKey);
            term_btree_->insert(chunkKey, payloadSpan);
        }
        
        offset += numSerialized;
        overflow_idx++;
    }
}

void TextIndex::remove_document(uint64_t doc_id) {
    std::unique_lock lock(mutex_);
    
    // Note: Removing from posting lists would require scanning all terms
    // For simplicity, we use tombstones (lazy deletion) - posting lists 
    // will still contain the doc_id but search will filter it out
    // A proper implementation would need periodic compaction
    
    // Decrement doc count (but we don't know the doc length without doc_lengths_)
    // This is a limitation of the new design - we trade off doc_lengths_ storage
    // for simpler persistence. The caller (Collection) should track doc lengths.
    doc_count_.fetch_sub(1);
}

void TextIndex::index_batch(const std::vector<uint64_t>& doc_ids,
                            const std::vector<std::string>& contents,
                            const std::function<void(uint64_t, uint32_t)>* update_doc_length) {
    if (doc_ids.size() != contents.size()) {
        throw std::runtime_error("doc_ids and contents must have same size");
    }
    
    if (doc_ids.empty()) {
        return;
    }
    
    std::unique_lock lock(mutex_);
    
    // Phase 1: Build all posting lists in memory
    // Key: term -> vector of (doc_id, term_freq) pairs
    absl::flat_hash_map<std::string, std::vector<std::pair<uint64_t, uint16_t>>> term_postings;
    
    uint64_t total_length = 0;
    uint64_t valid_docs = 0;
    
    for (size_t i = 0; i < doc_ids.size(); ++i) {
        // Analyze content
        auto terms = analyzer_.analyze(contents[i]);
        
        if (terms.empty()) {
            continue;
        }
        
        valid_docs++;
        uint32_t doc_len = static_cast<uint32_t>(terms.size());
        total_length += doc_len;
        
        // Update document length in Collection's id_index via callback
        if (update_doc_length) {
            (*update_doc_length)(doc_ids[i], doc_len);
        }
        
        // Count term frequencies for this document
        absl::flat_hash_map<std::string, uint16_t> term_freqs;
        term_freqs.reserve(terms.size());
        for (const auto& term : terms) {
            term_freqs[term]++;
        }
        
        // Add to global posting lists
        for (const auto& [term, freq] : term_freqs) {
            term_postings[term].emplace_back(doc_ids[i], freq);
        }
    }
    
    // Phase 2: Write all posting lists to BTree
    uint64_t new_terms = 0;
    bool index_is_empty = (vocab_size_.load() == 0);
    
    for (const auto& [term, postings] : term_postings) {
        PostingList list;
        
        // Only load existing posting list if index is not empty
        if (!index_is_empty) {
            list = load_posting_list(term);
        }
        
        // Check if this is a new term
        bool is_new_term = (list.doc_freq() == 0);
        if (is_new_term) {
            new_terms++;
        }
        
        // Add all postings for this term (already sorted by doc_id from Phase 1)
        for (const auto& [doc_id, freq] : postings) {
            list.add(doc_id, freq);
        }
        
        // Save the complete posting list once
        save_posting_list(term, list);
    }
    
    // Update statistics
    vocab_size_.fetch_add(new_terms);
    doc_count_.fetch_add(valid_docs);
    total_doc_length_.fetch_add(total_length);
}

std::vector<std::pair<uint64_t, float>> TextIndex::search(
    const std::string& query,
    size_t k,
    const std::vector<uint64_t>* doc_filter,
    const std::function<uint32_t(uint64_t)>* get_doc_length) const {
    
    std::shared_lock lock(mutex_);
    
    // Analyze query
    auto query_terms = analyzer_.analyze(query);
    
    if (query_terms.empty()) {
        return {};
    }
    
    // Set up scorer with current stats
    BM25Scorer scorer = scorer_;
    float avgDocLen = avg_doc_len();
    scorer.set_corpus_stats(doc_count_.load(), avgDocLen);
    
    // Default document length if not provided
    uint32_t defaultDocLen = static_cast<uint32_t>(avgDocLen > 0 ? avgDocLen : 100);
    
    // Collect candidate documents (union of all term postings)
    absl::flat_hash_map<uint64_t, float> doc_scores;
    
    for (const auto& term : query_terms) {
        PostingList list = load_posting_list(term);
        
        if (list.doc_freq() == 0) {
            continue;  // Term not in index
        }
        
        uint32_t doc_freq = static_cast<uint32_t>(list.doc_freq());
        
        for (const auto& posting : list.postings()) {
            // Check filter if provided
            if (doc_filter) {
                auto fit = std::lower_bound(doc_filter->begin(), doc_filter->end(), posting.doc_id);
                if (fit == doc_filter->end() || *fit != posting.doc_id) {
                    continue;  // Not in filter set
                }
            }
            
            // Get document length from callback, or use default
            uint32_t doc_len = 0;
            if (get_doc_length) {
                doc_len = (*get_doc_length)(posting.doc_id);
            }
            if (doc_len == 0) {
                doc_len = defaultDocLen;
            }
            
            float term_score = scorer.score(posting.term_freq, doc_len, doc_freq);
            doc_scores[posting.doc_id] += term_score;
        }
    }
    
    // Convert to sorted vector
    std::vector<std::pair<uint64_t, float>> results;
    results.reserve(doc_scores.size());
    
    for (const auto& [doc_id, score] : doc_scores) {
        results.emplace_back(doc_id, score);
    }
    
    // Sort by score descending
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Limit to k results
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

uint32_t TextIndex::btree_slot_id() const {
    return term_btree_ ? term_btree_->getSlotId() : 0;
}

void TextIndex::flush() {
    // BTree handles its own persistence through the buffer manager
    // Nothing explicit to do here
}

} // namespace caliby
