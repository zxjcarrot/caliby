/**
 * @file text_index.cpp
 * @brief Implementation of BM25 Text Index for Collection System
 */

#include "text_index.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <cctype>
#include <numeric>

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

void PostingList::add(uint64_t doc_id, uint16_t term_freq, const std::vector<uint16_t>& positions) {
    Posting posting{doc_id, term_freq, positions};
    
    // Fast path: if posting list is empty or doc_id > last, just append
    if (postings_.empty() || doc_id > postings_.back().doc_id) {
        postings_.push_back(std::move(posting));
        
        // Update skip pointers if needed
        if (postings_.size() % SKIP_INTERVAL == 0) {
            skip_pointers_.emplace_back(doc_id, postings_.size() - 1);
        }
        return;
    }
    
    // Slow path: need to find insertion point (rare during bulk insert)
    auto it = std::lower_bound(postings_.begin(), postings_.end(), posting,
        [](const Posting& a, const Posting& b) { return a.doc_id < b.doc_id; });
    
    if (it != postings_.end() && it->doc_id == doc_id) {
        // Update existing posting
        it->term_freq = term_freq;
        it->positions = positions;
    } else {
        postings_.insert(it, posting);
        
        // Update skip pointers if needed
        if (postings_.size() % SKIP_INTERVAL == 0) {
            skip_pointers_.emplace_back(doc_id, postings_.size() - 1);
        }
    }
}

std::vector<uint8_t> PostingList::serialize() const {
    std::vector<uint8_t> data;
    
    // Header: posting count (4 bytes)
    uint32_t count = static_cast<uint32_t>(postings_.size());
    data.push_back((count >> 24) & 0xFF);
    data.push_back((count >> 16) & 0xFF);
    data.push_back((count >> 8) & 0xFF);
    data.push_back(count & 0xFF);
    
    // Postings: doc_id (8 bytes) + term_freq (2 bytes) + position_count (2 bytes) + positions
    for (const auto& posting : postings_) {
        // doc_id (big-endian)
        for (int i = 7; i >= 0; --i) {
            data.push_back((posting.doc_id >> (i * 8)) & 0xFF);
        }
        
        // term_freq
        data.push_back((posting.term_freq >> 8) & 0xFF);
        data.push_back(posting.term_freq & 0xFF);
        
        // position count
        uint16_t pos_count = static_cast<uint16_t>(posting.positions.size());
        data.push_back((pos_count >> 8) & 0xFF);
        data.push_back(pos_count & 0xFF);
        
        // positions
        for (uint16_t pos : posting.positions) {
            data.push_back((pos >> 8) & 0xFF);
            data.push_back(pos & 0xFF);
        }
    }
    
    return data;
}

PostingList PostingList::deserialize(const uint8_t* data, size_t len) {
    PostingList list;
    
    if (len < 4) {
        return list;
    }
    
    // Read posting count
    uint32_t count = (static_cast<uint32_t>(data[0]) << 24) |
                     (static_cast<uint32_t>(data[1]) << 16) |
                     (static_cast<uint32_t>(data[2]) << 8) |
                     static_cast<uint32_t>(data[3]);
    
    size_t offset = 4;
    
    for (uint32_t i = 0; i < count && offset + 12 <= len; ++i) {
        Posting posting;
        
        // doc_id
        posting.doc_id = 0;
        for (int j = 0; j < 8; ++j) {
            posting.doc_id = (posting.doc_id << 8) | data[offset + j];
        }
        offset += 8;
        
        // term_freq
        posting.term_freq = (static_cast<uint16_t>(data[offset]) << 8) |
                           static_cast<uint16_t>(data[offset + 1]);
        offset += 2;
        
        // position count
        uint16_t pos_count = (static_cast<uint16_t>(data[offset]) << 8) |
                            static_cast<uint16_t>(data[offset + 1]);
        offset += 2;
        
        // positions
        posting.positions.reserve(pos_count);
        for (uint16_t p = 0; p < pos_count && offset + 2 <= len; ++p) {
            uint16_t pos = (static_cast<uint16_t>(data[offset]) << 8) |
                          static_cast<uint16_t>(data[offset + 1]);
            posting.positions.push_back(pos);
            offset += 2;
        }
        
        list.postings_.push_back(std::move(posting));
    }
    
    return list;
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
{
    // Get buffer manager
    if (bm_ptr) {
        bm_ = bm_ptr;
    }
}

std::unique_ptr<TextIndex> TextIndex::open(uint32_t index_id) {
    // For now, create with default settings - would need to read metadata
    auto index = std::make_unique<TextIndex>(0, index_id);
    index->load_metadata();
    return index;
}

TextIndex::~TextIndex() {
    try {
        flush();
    } catch (...) {
        // Ignore flush errors in destructor
    }
}

void TextIndex::index_document(uint64_t doc_id, const std::string& content) {
    std::unique_lock lock(mutex_);
    
    // Analyze content
    auto terms = analyzer_.analyze(content);
    
    if (terms.empty()) {
        return;
    }
    
    // Count term frequencies
    absl::flat_hash_map<std::string, uint16_t> term_freqs;
    term_freqs.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        term_freqs[terms[i]]++;
    }
    
    // Update document length and running total
    uint32_t doc_len = static_cast<uint32_t>(terms.size());
    doc_lengths_[doc_id] = doc_len;
    total_doc_length_.fetch_add(doc_len);
    
    // Add to posting lists
    for (const auto& [term, freq] : term_freqs) {
        auto it = term_dict_.find(term);
        if (it == term_dict_.end()) {
            it = term_dict_.emplace(term, PostingList()).first;
            vocab_size_.fetch_add(1);
        }
        it->second.add(doc_id, freq);
    }
    
    // Update stats incrementally - O(1) instead of O(n)
    uint64_t new_doc_count = doc_count_.fetch_add(1) + 1;
    avg_doc_len_.store(static_cast<float>(total_doc_length_.load()) / new_doc_count);
}

void TextIndex::remove_document(uint64_t doc_id) {
    std::unique_lock lock(mutex_);
    
    // Remove from doc_lengths
    auto len_it = doc_lengths_.find(doc_id);
    if (len_it == doc_lengths_.end()) {
        return;  // Document not indexed
    }
    
    // Decrement running total
    total_doc_length_.fetch_sub(len_it->second);
    doc_lengths_.erase(len_it);
    
    // Note: Removing from posting lists would require rebuild or lazy deletion
    // For simplicity, we just mark as deleted and leave in posting lists
    // A proper implementation would use tombstones and periodic compaction
    
    doc_count_.fetch_sub(1);
    update_statistics();
}

void TextIndex::index_batch(const std::vector<uint64_t>& doc_ids,
                            const std::vector<std::string>& contents) {
    if (doc_ids.size() != contents.size()) {
        throw std::runtime_error("doc_ids and contents must have same size");
    }
    
    for (size_t i = 0; i < doc_ids.size(); ++i) {
        index_document(doc_ids[i], contents[i]);
    }
}

std::vector<std::pair<uint64_t, float>> TextIndex::search(
    const std::string& query,
    size_t k,
    const std::vector<uint64_t>* doc_filter) const {
    
    std::shared_lock lock(mutex_);
    
    // Analyze query
    auto query_terms = analyzer_.analyze(query);
    
    if (query_terms.empty()) {
        return {};
    }
    
    // Set up scorer with current stats
    BM25Scorer scorer = scorer_;
    scorer.set_corpus_stats(doc_count_.load(), avg_doc_len_.load());
    
    // Collect candidate documents (union of all term postings)
    absl::flat_hash_map<uint64_t, float> doc_scores;
    
    for (const auto& term : query_terms) {
        auto it = term_dict_.find(term);
        if (it == term_dict_.end()) {
            continue;  // Term not in index
        }
        
        const auto& posting_list = it->second;
        uint32_t doc_freq = static_cast<uint32_t>(posting_list.doc_freq());
        
        for (const auto& posting : posting_list.postings()) {
            // Check filter if provided
            if (doc_filter) {
                auto fit = std::lower_bound(doc_filter->begin(), doc_filter->end(), posting.doc_id);
                if (fit == doc_filter->end() || *fit != posting.doc_id) {
                    continue;  // Not in filter set
                }
            }
            
            // Get document length
            auto len_it = doc_lengths_.find(posting.doc_id);
            if (len_it == doc_lengths_.end()) {
                continue;  // Document was deleted
            }
            
            float term_score = scorer.score(posting.term_freq, len_it->second, doc_freq);
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

uint64_t TextIndex::vocab_size() const {
    return vocab_size_.load();
}

uint64_t TextIndex::doc_count() const {
    return doc_count_.load();
}

float TextIndex::avg_doc_len() const {
    return avg_doc_len_.load();
}

void TextIndex::flush() {
    std::unique_lock lock(mutex_);
    save_metadata();
}

void TextIndex::update_statistics() {
    // Now uses incremental updates via total_doc_length_
    // This function is kept for compatibility but the main work
    // is done in index_document() and remove_document()
    uint64_t count = doc_count_.load();
    if (count == 0) {
        avg_doc_len_.store(0.0f);
        return;
    }
    avg_doc_len_.store(static_cast<float>(total_doc_length_.load()) / count);
}

void TextIndex::load_metadata() {
    // TODO: Load from disk using buffer manager
    // For now, index is in-memory only
}

void TextIndex::save_metadata() {
    // TODO: Save to disk using buffer manager
    // For now, index is in-memory only
}

} // namespace caliby
