/**
 * @file collection.cpp
 * @brief Implementation of Caliby Collection System
 */

#include "collection.hpp"
#include "btree_index.hpp"
#include "text_index.hpp"
#include "catalog.hpp"
#include "hnsw.hpp"
#include "distance.hpp"
#include "logging.hpp"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <stdexcept>

// Type aliases for HNSW indices with different distance metrics
using HnswL2 = HNSW<hnsw_distance::SIMDAcceleratedL2>;
using HnswIP = HNSW<hnsw_distance::SIMDAcceleratedIP>;
using HnswCosine = HNSW<hnsw_distance::SIMDAcceleratedCosine>;

namespace caliby {

//=============================================================================
// Concrete HNSW Wrapper Implementations
//=============================================================================

/**
 * Template wrapper for HNSW indices that implements HNSWIndexBase interface.
 */
template<typename DistanceType, DistanceMetric MetricType>
class HNSWIndexImpl : public HNSWIndexBase {
public:
    HNSWIndexImpl(size_t max_elements, size_t dim, size_t M, size_t ef_construction,
                  bool enable_prefetch = true, bool skip_recovery = false,
                  uint32_t index_id = 0, const std::string& name = "")
        : hnsw_(max_elements, dim, M, ef_construction, enable_prefetch, skip_recovery, index_id, name)
        , metric_type_(MetricType) {}
    
    void addPointWithId(const float* data, uint32_t id) override {
        hnsw_.addPointWithId(data, id);
    }
    
    void addPointsWithIdsParallel(const std::vector<const float*>& data_ptrs,
                                   const std::vector<uint32_t>& ids,
                                   size_t num_threads = 0) override {
        hnsw_.addPointsWithIdsParallel(data_ptrs, ids, num_threads);
    }
    
    std::vector<std::pair<float, uint32_t>> searchKnn(
        const float* query, size_t k, size_t ef_search = 100) override {
        return hnsw_.searchKnn(query, k, ef_search);
    }
    
    DistanceMetric metric() const override {
        return metric_type_;
    }
    
    bool wasRecovered() const override {
        return hnsw_.wasRecovered();
    }
    
    // Access underlying HNSW for operations not in base interface
    HNSW<DistanceType>& underlying() { return hnsw_; }
    const HNSW<DistanceType>& underlying() const { return hnsw_; }

private:
    HNSW<DistanceType> hnsw_;
    DistanceMetric metric_type_;
};

// Type aliases for concrete implementations
using HNSWIndexL2 = HNSWIndexImpl<hnsw_distance::SIMDAcceleratedL2, DistanceMetric::L2>;
using HNSWIndexIP = HNSWIndexImpl<hnsw_distance::SIMDAcceleratedIP, DistanceMetric::IP>;
using HNSWIndexCosine = HNSWIndexImpl<hnsw_distance::SIMDAcceleratedCosine, DistanceMetric::COSINE>;

/**
 * Factory function to create HNSW index with appropriate distance metric.
 */
std::unique_ptr<HNSWIndexBase> createHNSWIndex(
    DistanceMetric metric,
    size_t max_elements, size_t dim, size_t M, size_t ef_construction,
    bool enable_prefetch = true, bool skip_recovery = false,
    uint32_t index_id = 0, const std::string& name = "") {
    
    switch (metric) {
        case DistanceMetric::L2:
            return std::make_unique<HNSWIndexL2>(
                max_elements, dim, M, ef_construction, enable_prefetch, skip_recovery, index_id, name);
        case DistanceMetric::IP:
            return std::make_unique<HNSWIndexIP>(
                max_elements, dim, M, ef_construction, enable_prefetch, skip_recovery, index_id, name);
        case DistanceMetric::COSINE:
            return std::make_unique<HNSWIndexCosine>(
                max_elements, dim, M, ef_construction, enable_prefetch, skip_recovery, index_id, name);
        default:
            throw std::runtime_error("Unknown distance metric");
    }
}

//=============================================================================
// Schema Implementation
//=============================================================================

void Schema::add_field(const std::string& name, FieldType type, bool nullable) {
    if (has_field(name)) {
        throw std::runtime_error("Field already exists: " + name);
    }
    name_to_index_[name] = fields_.size();
    fields_.emplace_back(name, type, nullable);
}

const FieldDef* Schema::get_field(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        return nullptr;
    }
    return &fields_[it->second];
}

bool Schema::has_field(const std::string& name) const {
    return name_to_index_.find(name) != name_to_index_.end();
}

bool Schema::validate(const nlohmann::json& metadata, std::string& error) const {
    for (const auto& field : fields_) {
        auto it = metadata.find(field.name);
        
        if (it == metadata.end()) {
            if (!field.nullable) {
                error = "Missing required field: " + field.name;
                return false;
            }
            continue;
        }
        
        const auto& val = *it;
        bool type_ok = false;
        
        switch (field.type) {
            case FieldType::STRING:
                type_ok = val.is_string();
                break;
            case FieldType::INT:
                type_ok = val.is_number_integer();
                break;
            case FieldType::FLOAT:
                type_ok = val.is_number();
                break;
            case FieldType::BOOL:
                type_ok = val.is_boolean();
                break;
            case FieldType::STRING_ARRAY:
                type_ok = val.is_array() && std::all_of(val.begin(), val.end(),
                    [](const nlohmann::json& v) { return v.is_string(); });
                break;
            case FieldType::INT_ARRAY:
                type_ok = val.is_array() && std::all_of(val.begin(), val.end(),
                    [](const nlohmann::json& v) { return v.is_number_integer(); });
                break;
        }
        
        if (!type_ok) {
            error = "Invalid type for field: " + field.name;
            return false;
        }
    }
    return true;
}

nlohmann::json Schema::to_json() const {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& field : fields_) {
        nlohmann::json f;
        f["name"] = field.name;
        f["nullable"] = field.nullable;
        switch (field.type) {
            case FieldType::STRING: f["type"] = "string"; break;
            case FieldType::INT: f["type"] = "int"; break;
            case FieldType::FLOAT: f["type"] = "float"; break;
            case FieldType::BOOL: f["type"] = "bool"; break;
            case FieldType::STRING_ARRAY: f["type"] = "string[]"; break;
            case FieldType::INT_ARRAY: f["type"] = "int[]"; break;
        }
        j.push_back(f);
    }
    return j;
}

Schema Schema::from_json(const nlohmann::json& j) {
    Schema schema;
    for (const auto& f : j) {
        std::string name = f["name"];
        std::string type_str = f["type"];
        bool nullable = f.value("nullable", true);
        
        FieldType type;
        if (type_str == "string") type = FieldType::STRING;
        else if (type_str == "int") type = FieldType::INT;
        else if (type_str == "float") type = FieldType::FLOAT;
        else if (type_str == "bool") type = FieldType::BOOL;
        else if (type_str == "string[]") type = FieldType::STRING_ARRAY;
        else if (type_str == "int[]") type = FieldType::INT_ARRAY;
        else throw std::runtime_error("Unknown field type: " + type_str);
        
        schema.add_field(name, type, nullable);
    }
    return schema;
}

Schema Schema::from_dict(const std::unordered_map<std::string, std::string>& dict) {
    Schema schema;
    for (const auto& [name, type_str] : dict) {
        FieldType type;
        if (type_str == "string") type = FieldType::STRING;
        else if (type_str == "int") type = FieldType::INT;
        else if (type_str == "float") type = FieldType::FLOAT;
        else if (type_str == "bool") type = FieldType::BOOL;
        else if (type_str == "string[]") type = FieldType::STRING_ARRAY;
        else if (type_str == "int[]") type = FieldType::INT_ARRAY;
        else throw std::runtime_error("Unknown field type: " + type_str);
        
        schema.add_field(name, type, true);
    }
    return schema;
}

//=============================================================================
// MetadataValue Helpers
//=============================================================================

FieldType get_field_type(const MetadataValue& value) {
    return std::visit([](auto&& v) -> FieldType {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) return FieldType::STRING;
        else if constexpr (std::is_same_v<T, std::string>) return FieldType::STRING;
        else if constexpr (std::is_same_v<T, int64_t>) return FieldType::INT;
        else if constexpr (std::is_same_v<T, double>) return FieldType::FLOAT;
        else if constexpr (std::is_same_v<T, bool>) return FieldType::BOOL;
        else if constexpr (std::is_same_v<T, std::vector<std::string>>) return FieldType::STRING_ARRAY;
        else if constexpr (std::is_same_v<T, std::vector<int64_t>>) return FieldType::INT_ARRAY;
        else return FieldType::STRING;
    }, value);
}

//=============================================================================
// FilterCondition Implementation
//=============================================================================

FilterCondition FilterCondition::from_json(const nlohmann::json& j) {
    FilterCondition cond;
    
    if (j.contains("$and")) {
        cond.op = FilterOp::AND;
        for (const auto& child : j["$and"]) {
            cond.children.push_back(from_json(child));
        }
        return cond;
    }
    
    if (j.contains("$or")) {
        cond.op = FilterOp::OR;
        for (const auto& child : j["$or"]) {
            cond.children.push_back(from_json(child));
        }
        return cond;
    }
    
    // Single field condition
    for (auto& [key, val] : j.items()) {
        cond.field = key;
        
        if (val.is_object()) {
            // Operator form: {"field": {"$gt": 10}}
            for (auto& [op_str, op_val] : val.items()) {
                if (op_str == "$eq") cond.op = FilterOp::EQ;
                else if (op_str == "$ne") cond.op = FilterOp::NE;
                else if (op_str == "$gt") cond.op = FilterOp::GT;
                else if (op_str == "$gte") cond.op = FilterOp::GTE;
                else if (op_str == "$lt") cond.op = FilterOp::LT;
                else if (op_str == "$lte") cond.op = FilterOp::LTE;
                else if (op_str == "$in") cond.op = FilterOp::IN;
                else if (op_str == "$nin") cond.op = FilterOp::NIN;
                else if (op_str == "$contains") cond.op = FilterOp::CONTAINS;
                else throw std::runtime_error("Unknown filter operator: " + op_str);
                
                // Convert JSON value to MetadataValue
                if (op_val.is_string()) {
                    cond.value = op_val.get<std::string>();
                } else if (op_val.is_number_integer()) {
                    cond.value = op_val.get<int64_t>();
                } else if (op_val.is_number_float()) {
                    cond.value = op_val.get<double>();
                } else if (op_val.is_boolean()) {
                    cond.value = op_val.get<bool>();
                } else if (op_val.is_array()) {
                    // Check first element type
                    if (!op_val.empty() && op_val[0].is_string()) {
                        cond.value = op_val.get<std::vector<std::string>>();
                    } else {
                        cond.value = op_val.get<std::vector<int64_t>>();
                    }
                }
                break;  // Only one operator per field
            }
        } else {
            // Implicit equality: {"field": "value"}
            cond.op = FilterOp::EQ;
            if (val.is_string()) {
                cond.value = val.get<std::string>();
            } else if (val.is_number_integer()) {
                cond.value = val.get<int64_t>();
            } else if (val.is_number_float()) {
                cond.value = val.get<double>();
            } else if (val.is_boolean()) {
                cond.value = val.get<bool>();
            }
        }
        break;  // Only one field per condition object
    }
    
    return cond;
}

bool FilterCondition::evaluate(const nlohmann::json& metadata) const {
    switch (op) {
        case FilterOp::AND: {
            for (const auto& child : children) {
                if (!child.evaluate(metadata)) return false;
            }
            return true;
        }
        case FilterOp::OR: {
            for (const auto& child : children) {
                if (child.evaluate(metadata)) return true;
            }
            return false;
        }
        default: break;
    }
    
    // Field-level comparison
    auto it = metadata.find(field);
    if (it == metadata.end()) {
        return false;  // Field not present
    }
    
    const auto& doc_val = *it;
    
    // Helper to compare values
    auto compare = [&](auto&& expected) -> int {
        using T = std::decay_t<decltype(expected)>;
        if constexpr (std::is_same_v<T, std::string>) {
            if (!doc_val.is_string()) return -2;  // Type mismatch
            return doc_val.get<std::string>().compare(expected);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (!doc_val.is_number()) return -2;
            int64_t dv = doc_val.get<int64_t>();
            return (dv < expected) ? -1 : (dv > expected) ? 1 : 0;
        } else if constexpr (std::is_same_v<T, double>) {
            if (!doc_val.is_number()) return -2;
            double dv = doc_val.get<double>();
            return (dv < expected) ? -1 : (dv > expected) ? 1 : 0;
        } else if constexpr (std::is_same_v<T, bool>) {
            if (!doc_val.is_boolean()) return -2;
            return doc_val.get<bool>() == expected ? 0 : 1;
        }
        return -2;
    };
    
    switch (op) {
        case FilterOp::EQ:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) return false;
                else return compare(v) == 0;
            }, value);
            
        case FilterOp::NE:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) return true;
                else return compare(v) != 0;
            }, value);
            
        case FilterOp::GT:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) return false;
                else if constexpr (std::is_same_v<T, std::vector<std::string>> || 
                                   std::is_same_v<T, std::vector<int64_t>>) return false;
                else return compare(v) > 0;
            }, value);
            
        case FilterOp::GTE:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) return false;
                else if constexpr (std::is_same_v<T, std::vector<std::string>> || 
                                   std::is_same_v<T, std::vector<int64_t>>) return false;
                else return compare(v) >= 0;
            }, value);
            
        case FilterOp::LT:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) return false;
                else if constexpr (std::is_same_v<T, std::vector<std::string>> || 
                                   std::is_same_v<T, std::vector<int64_t>>) return false;
                else {
                    int cmp = compare(v);
                    return cmp != -2 && cmp < 0;
                }
            }, value);
            
        case FilterOp::LTE:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) return false;
                else if constexpr (std::is_same_v<T, std::vector<std::string>> || 
                                   std::is_same_v<T, std::vector<int64_t>>) return false;
                else {
                    int cmp = compare(v);
                    return cmp != -2 && cmp <= 0;
                }
            }, value);
            
        case FilterOp::IN:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    if (!doc_val.is_string()) return false;
                    std::string dv = doc_val.get<std::string>();
                    return std::find(v.begin(), v.end(), dv) != v.end();
                } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                    if (!doc_val.is_number_integer()) return false;
                    int64_t dv = doc_val.get<int64_t>();
                    return std::find(v.begin(), v.end(), dv) != v.end();
                }
                return false;
            }, value);
            
        case FilterOp::NIN:
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    if (!doc_val.is_string()) return true;
                    std::string dv = doc_val.get<std::string>();
                    return std::find(v.begin(), v.end(), dv) == v.end();
                } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                    if (!doc_val.is_number_integer()) return true;
                    int64_t dv = doc_val.get<int64_t>();
                    return std::find(v.begin(), v.end(), dv) == v.end();
                }
                return true;
            }, value);
            
        case FilterOp::CONTAINS:
            // Array contains check
            if (!doc_val.is_array()) return false;
            return std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    for (const auto& elem : doc_val) {
                        if (elem.is_string() && elem.get<std::string>() == v) return true;
                    }
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    for (const auto& elem : doc_val) {
                        if (elem.is_number_integer() && elem.get<int64_t>() == v) return true;
                    }
                }
                return false;
            }, value);
            
        default:
            return false;
    }
}

//=============================================================================
// Collection Implementation
//=============================================================================

Collection::Collection(const std::string& name,
                       const Schema& schema,
                       uint32_t vector_dim,
                       DistanceMetric distance_metric)
    : name_(name)
    , schema_(schema)
    , vector_dim_(vector_dim)
    , distance_metric_(distance_metric)
    , collection_id_(0)
{
    // Get buffer manager from global
    bm_ = bm_ptr;
    if (!bm_) {
        throw std::runtime_error("Buffer manager not initialized. Call caliby.open() first.");
    }
    
    // Create collection through catalog
    IndexCatalog& catalog = IndexCatalog::instance();
    if (!catalog.is_initialized()) {
        throw std::runtime_error("Catalog not initialized. Call caliby.open() first.");
    }
    
    // Create collection entry in catalog
    IndexConfig config;
    config.dimensions = vector_dim;
    config.max_elements = 0;  // Collections don't have a fixed max
    
    IndexHandle handle = catalog.create_index(name, IndexType::COLLECTION, config);
    collection_id_ = handle.index_id();
    file_fd_ = handle.file_fd();
    
    // Initialize collection metadata page
    save_metadata();
    
    CALIBY_LOG_INFO("Collection", "Created collection '", name, "' (id=", collection_id_, ")");
}

std::unique_ptr<Collection> Collection::open(const std::string& name) {
    IndexCatalog& catalog = IndexCatalog::instance();
    if (!catalog.is_initialized()) {
        throw std::runtime_error("Catalog not initialized. Call caliby.open() first.");
    }
    
    IndexHandle handle = catalog.open_index(name);
    if (handle.type() != IndexType::COLLECTION) {
        throw std::runtime_error("'" + name + "' is not a collection");
    }
    
    // Create collection object and load metadata
    auto collection = std::unique_ptr<Collection>(new Collection());
    collection->name_ = name;
    collection->collection_id_ = handle.index_id();
    collection->file_fd_ = handle.file_fd();
    collection->bm_ = bm_ptr;
    
    // load_metadata() now also recovers the DocIdIndex from its persisted BTree slotId
    collection->load_metadata();
    
    // Note: rebuild_id_index() is no longer needed since DocIdIndex is now
    // properly recovered from the persisted BTree using its slotId
    
    // Recover indices that belong to this collection
    // Indices are named: collection_name + "_" + index_name
    std::string prefix = name + "_";
    auto all_indexes = catalog.list_indexes();
    
    for (const auto& idx_info : all_indexes) {
        // Check if this index belongs to our collection
        if (idx_info.name.substr(0, prefix.length()) == prefix) {
            std::string idx_name = idx_info.name.substr(prefix.length());
            
            if (idx_info.type == IndexType::HNSW) {
                // Get the stored HNSW config from catalog
                HNSWConfig hnsw_config = catalog.get_hnsw_config(idx_info.name);
                
                // Recover HNSW index
                IndexHandle idx_handle = catalog.open_index(idx_info.name);
                
                // Create HNSW object with correct params for recovery using collection's distance metric
                auto hnsw = createHNSWIndex(
                    collection->distance_metric_,  // Use collection's distance metric
                    1000000,                  // max_elements
                    collection->vector_dim_,  // dim
                    hnsw_config.M,            // M from catalog
                    hnsw_config.ef_construction, // ef_construction from catalog
                    true,                     // enable_prefetch
                    false,                    // skip_recovery (allow recovery)
                    idx_handle.index_id(),    // index_id
                    idx_info.name             // name
                );
                
                // Track index
                CollectionIndexInfo info;
                info.index_id = idx_handle.index_id();
                info.name = idx_name;
                info.type = "hnsw";
                info.status = "ready";
                info.config = {{"M", hnsw_config.M}, {"ef_construction", hnsw_config.ef_construction}};
                
                collection->indices_[idx_name] = info;
                collection->hnsw_indices_[idx_name] = std::move(hnsw);
                
                CALIBY_LOG_INFO("Collection", "Recovered HNSW index '", idx_name, 
                                "' for collection '", name, "'");
            }
            else if (idx_info.type == IndexType::TEXT) {
                // Get the stored text config from catalog
                TextTypeMetadata text_config = catalog.get_text_config(idx_info.name);
                
                CALIBY_LOG_DEBUG("Collection", "Recovery: text_config for '", idx_info.name, 
                                 "': btree_slot=", text_config.btree_slot_id,
                                 ", vocab=", text_config.vocab_size,
                                 ", docs=", text_config.doc_count);
                
                // Recover text index - open the handle first
                IndexHandle idx_handle = catalog.open_index(idx_info.name);
                
                std::unique_ptr<TextIndex> text_index;
                
                // Check if we have a valid BTree slot ID (persistent state)
                if (text_config.has_valid_btree()) {
                    // Recover from persistent BTree
                    text_index = TextIndex::open(
                        idx_handle.index_id(),
                        text_config.btree_slot_id,
                        text_config.vocab_size,
                        text_config.doc_count,
                        text_config.total_doc_length,
                        text_config.k1,
                        text_config.b
                    );
                    CALIBY_LOG_INFO("Collection", "Recovered text index from BTree slot ", text_config.btree_slot_id, 
                                    " with ", text_config.vocab_size, " terms, ", 
                                    text_config.doc_count, " docs");
                } else {
                    CALIBY_LOG_INFO("Collection", "No valid BTree slot for text index, rebuilding...");
                    // No valid BTree - need to rebuild from documents
                    AnalyzerType analyzer_type = AnalyzerType::STANDARD;
                    std::string analyzer_str(text_config.analyzer);
                    if (analyzer_str == "whitespace") {
                        analyzer_type = AnalyzerType::WHITESPACE;
                    } else if (analyzer_str == "none") {
                        analyzer_type = AnalyzerType::NONE;
                    }
                    
                    text_index = std::make_unique<TextIndex>(
                        collection->collection_id_,
                        idx_handle.index_id(),
                        analyzer_type,
                        std::string(text_config.language),
                        text_config.k1,
                        text_config.b
                    );
                    
                    // Callback to update doc length in id_index
                    Collection* coll_ptr = collection.get();
                    std::function<void(uint64_t, uint32_t)> update_doc_length = 
                        [coll_ptr](uint64_t doc_id, uint32_t doc_len) {
                            coll_ptr->id_index_update_doc_length(doc_id, doc_len);
                        };
                    
                    // Re-index existing documents (doc IDs start at 0)
                    uint64_t current_doc_count = collection->doc_count_.load();
                    for (uint64_t doc_id = 0; doc_id < current_doc_count; ++doc_id) {
                        try {
                            Document doc = collection->read_document(doc_id);
                            if (!doc.content.empty()) {
                                text_index->index_document(doc_id, doc.content, &update_doc_length);
                            }
                        } catch (...) {
                            // Document may not exist (deleted), skip
                        }
                    }
                }
                
                // Track index
                CollectionIndexInfo info;
                info.index_id = idx_handle.index_id();
                info.name = idx_name;
                info.type = "text";
                info.status = "ready";
                info.config = {
                    {"analyzer", std::string(text_config.analyzer)},
                    {"language", std::string(text_config.language)},
                    {"k1", text_config.k1},
                    {"b", text_config.b}
                };
                
                collection->indices_[idx_name] = info;
                collection->text_indices_[idx_name] = std::move(text_index);
                
                // Save updated BTree state if we rebuilt
                if (text_config.btree_slot_id == 0) {
                    collection->save_text_index_state(idx_name);
                }
                
                CALIBY_LOG_INFO("Collection", "Recovered text index '", idx_name, 
                                "' for collection '", name, "'");
            }
            else if (idx_info.type == IndexType::BTREE) {
                // Get the stored btree config from catalog
                BTreeTypeMetadata btree_config = catalog.get_btree_config(idx_info.name);
                
                // Recover btree index - open the handle first
                IndexHandle idx_handle = catalog.open_index(idx_info.name);
                
                // Get fields from config
                std::vector<std::string> fields = btree_config.get_fields();
                
                // Track index
                CollectionIndexInfo info;
                info.index_id = idx_handle.index_id();
                info.name = idx_name;
                info.type = "btree";
                info.status = "ready";
                
                nlohmann::json fields_json = nlohmann::json::array();
                for (const auto& field : fields) {
                    fields_json.push_back(field);
                }
                info.config = {
                    {"fields", fields_json},
                    {"unique", btree_config.unique}
                };
                if (fields.size() == 1) {
                    info.config["field"] = fields[0];
                }
                
                collection->indices_[idx_name] = info;
                
                CALIBY_LOG_INFO("Collection", "Recovered btree index '", idx_name, 
                                "' for collection '", name, "'");
            }
        }
    }
    
    return collection;
}

Collection::~Collection() {
    // Flush any pending changes
    // Check if system is still valid:
    // - bm_ptr may be null after shutdown_system()
    // - system_closed is true after close() but before shutdown_system()
    //   (index arrays are unregistered but bm_ptr is still valid)
    if (bm_ptr != nullptr && !system_closed) {
        flush();
    }
}

uint64_t Collection::doc_count() const {
    return doc_count_.load();
}

std::vector<uint64_t> Collection::add(const std::vector<std::string>& contents,
                                      const std::vector<nlohmann::json>& metadatas,
                                      const std::vector<std::vector<float>>& vectors) {
    
    if (contents.size() != metadatas.size()) {
        throw std::runtime_error("contents and metadatas must have same length");
    }
    if (!vectors.empty()) {
        if (vector_dim_ == 0) {
            throw std::runtime_error("Collection does not support vectors");
        }
        if (vectors.size() != contents.size()) {
            throw std::runtime_error("vectors must be same length as contents");
        }
    }
    
    // Assign IDs atomically without holding main lock
    std::vector<uint64_t> assigned_ids;
    assigned_ids.reserve(contents.size());
    uint64_t first_id = next_doc_id_.fetch_add(contents.size());
    for (size_t i = 0; i < contents.size(); ++i) {
        assigned_ids.push_back(first_id + i);
    }
    
    // Validate all metadata first (no lock needed)
    for (size_t i = 0; i < contents.size(); ++i) {
        std::string error;
        if (!schema_.validate(metadatas[i], error)) {
            throw std::runtime_error("Metadata validation failed for doc " + 
                                     std::to_string(assigned_ids[i]) + ": " + error);
        }
    }
    
    // Write documents - each write_document handles its own locking via id_index_mutex_
    // The slotted page uses GuardX for page-level locking
    {
        std::unique_lock lock(mutex_);  // Still need lock for doc_pages_head_/tail_ access
        for (size_t i = 0; i < contents.size(); ++i) {
            Document doc;
            doc.id = assigned_ids[i];
            doc.content = contents[i];
            doc.metadata = metadatas[i];
            write_document(doc);
        }
        doc_count_.fetch_add(contents.size());
    }
    
    // Add vectors to HNSW indices using parallel batch insert
    if (!vectors.empty()) {
        std::shared_lock lock(mutex_);  // Only need shared lock to read hnsw_indices_
        
        // Prepare data pointers and IDs for batch insert
        std::vector<const float*> data_ptrs;
        std::vector<uint32_t> node_ids;
        data_ptrs.reserve(vectors.size());
        node_ids.reserve(vectors.size());
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            data_ptrs.push_back(vectors[i].data());
            node_ids.push_back(static_cast<uint32_t>(assigned_ids[i]));
        }
        
        for (auto& [name, hnsw] : hnsw_indices_) {
            if (hnsw) {
                // Use parallel batch insert for better performance
                hnsw->addPointsWithIdsParallel(data_ptrs, node_ids);
            }
        }
    }
    
    // Index text content - TextIndex has its own internal locking
    {
        std::shared_lock lock(mutex_);  // Only need shared lock to read text_indices_
        
        // Callback to update doc length in id_index
        std::function<void(uint64_t, uint32_t)> update_doc_length = 
            [this](uint64_t doc_id, uint32_t doc_len) {
                this->id_index_update_doc_length(doc_id, doc_len);
            };
        
        for (auto& [name, text_index] : text_indices_) {
            if (text_index) {
                // Use batch indexing for O(n) instead of O(n²)
                text_index->index_batch(assigned_ids, contents, &update_doc_length);
            }
        }
    }
    
    // Save metadata after each add for durability
    // This ensures doc_count, id_index btree slot, and page chain are persisted
    {
        std::unique_lock lock(mutex_);
        save_metadata();
        save_all_text_index_states();  // Persist TextIndex BTree state
    }
    
    return assigned_ids;
}

std::vector<Document> Collection::get(const std::vector<uint64_t>& ids) {
    std::shared_lock lock(mutex_);
    
    std::vector<Document> results;
    results.reserve(ids.size());
    
    for (uint64_t id : ids) {
        try {
            results.push_back(read_document(id));
        } catch (const std::exception& e) {
            // Document not found - add empty doc with just ID
            Document doc;
            doc.id = id;
            results.push_back(doc);
        }
    }
    
    return results;
}

std::vector<Document> Collection::get(const FilterCondition& where,
                                       size_t limit,
                                       size_t offset) {
    std::shared_lock lock(mutex_);
    
    std::vector<Document> results;
    
    // Evaluate filter to get matching doc IDs
    std::vector<uint64_t> matching_ids = evaluate_filter(where);
    
    // Apply offset and limit
    size_t start = std::min(offset, matching_ids.size());
    size_t end = std::min(start + limit, matching_ids.size());
    
    for (size_t i = start; i < end; ++i) {
        try {
            results.push_back(read_document(matching_ids[i]));
        } catch (...) {
            // Skip documents that can't be read
        }
    }
    
    return results;
}

void Collection::update(const std::vector<uint64_t>& ids,
                        const std::vector<nlohmann::json>& metadatas) {
    if (ids.size() != metadatas.size()) {
        throw std::runtime_error("ids and metadatas must have same length");
    }
    
    std::unique_lock lock(mutex_);
    
    for (size_t i = 0; i < ids.size(); ++i) {
        // Read existing document
        Document doc = read_document(ids[i]);
        
        // Merge metadata (partial update)
        for (auto& [key, val] : metadatas[i].items()) {
            doc.metadata[key] = val;
        }
        
        // Validate updated metadata
        std::string error;
        if (!schema_.validate(doc.metadata, error)) {
            throw std::runtime_error("Metadata validation failed for doc " + 
                                     std::to_string(ids[i]) + ": " + error);
        }
        
        // Delete old and write new
        delete_document_internal(ids[i]);
        write_document(doc);
    }
}

void Collection::delete_docs(const std::vector<uint64_t>& ids) {
    std::unique_lock lock(mutex_);
    
    for (uint64_t id : ids) {
        delete_document_internal(id);
        doc_count_.fetch_sub(1);
    }
    
    save_metadata();
}

size_t Collection::delete_docs(const FilterCondition& where) {
    std::unique_lock lock(mutex_);
    
    std::vector<uint64_t> matching_ids = evaluate_filter(where);
    
    for (uint64_t id : matching_ids) {
        delete_document_internal(id);
    }
    
    doc_count_.fetch_sub(matching_ids.size());
    save_metadata();
    
    return matching_ids.size();
}

//=============================================================================
// Index Operations
//=============================================================================

void Collection::create_hnsw_index(const std::string& name, size_t M, size_t ef_construction) {
    if (vector_dim_ == 0) {
        throw std::runtime_error("Collection does not support vectors");
    }
    
    std::unique_lock lock(mutex_);
    
    // Check if index already exists in memory (may have been recovered on open)
    if (indices_.find(name) != indices_.end()) {
        // Index already exists - this is fine, just return silently
        // (supports idempotent create_index pattern)
        return;
    }
    
    // Check catalog for existing index (for recovery case)
    IndexCatalog& catalog = IndexCatalog::instance();
    std::string full_name = name_ + "_" + name;
    
    IndexHandle handle;
    bool recovering = false;
    
    if (catalog.index_exists(full_name)) {
        // Index exists in catalog - this is a recovery scenario
        handle = catalog.open_index(full_name);
        recovering = true;
        CALIBY_LOG_INFO("Collection", "Recovering HNSW index '", name, "' from catalog");
    } else {
        // Create new index through catalog
        handle = catalog.create_hnsw_index(
            full_name,  // Prefix with collection name
            vector_dim_,
            1000000,  // max_elements - will grow as needed
            M,
            ef_construction
        );
    }
    
    // Create actual HNSW object using collection's distance metric (will recover if data exists)
    auto hnsw = createHNSWIndex(
        distance_metric_,  // Use collection's distance metric
        1000000,        // max_elements
        vector_dim_,    // dim
        M,              // M
        ef_construction, // ef_construction
        true,           // enable_prefetch
        false,          // skip_recovery (allow recovery)
        handle.index_id(), // index_id
        full_name       // name
    );
    
    // Track index
    CollectionIndexInfo info;
    info.index_id = handle.index_id();
    info.name = name;
    info.type = "hnsw";
    info.status = "ready";
    info.config = {{"M", M}, {"ef_construction", ef_construction}};
    
    indices_[name] = info;
    hnsw_indices_[name] = std::move(hnsw);
    
    // If collection already has documents with vectors, populate the index
    // (This handles the case where create_hnsw_index is called after add())
    if (!recovering && doc_count_.load() > 0 && vector_dim_ > 0) {
        CALIBY_LOG_INFO("Collection", "Populating HNSW index '", name, 
                        "' with existing vectors...");
        
        uint64_t populated = 0;
        uint64_t total_docs = doc_count_.load();
        
        // Iterate through all documents and add their vectors
        // Note: We need to release the lock temporarily for read_document
        lock.unlock();
        
        for (uint64_t doc_id = 0; doc_id < next_doc_id_.load(); ++doc_id) {
            try {
                // Try to read the document to check if it exists
                // (some IDs might be deleted)
                Document doc = read_document(doc_id);
                
                // Get vector for this document from document metadata
                // The vector should be stored when the document was added
                // Actually, vectors are NOT stored in documents - they go directly to HNSW
                // So we can't recover them this way. We need to store vectors separately.
                
                // For now, we can only warn and continue
                // This is a limitation - we'd need to store vectors to support late index creation
                populated++;
            } catch (...) {
                // Document doesn't exist or error reading
            }
        }
        
        lock.lock();
        
        // Since vectors aren't stored separately, we can only populate if recovering
        // For now, warn the user that late index creation won't include existing vectors
        if (populated > 0) {
            CALIBY_LOG_WARN("Collection", "HNSW index '", name, 
                           "' created after documents were added. ",
                           "Vectors for existing ", populated, 
                           " documents are NOT indexed! ",
                           "Create the index BEFORE adding documents for best results.");
        }
    }
    
    if (recovering) {
        CALIBY_LOG_INFO("Collection", "Recovered HNSW index '", name, "' on collection '", name_, "'");
    } else {
        CALIBY_LOG_INFO("Collection", "Created HNSW index '", name, "' on collection '", name_, "'");
    }
}

void Collection::create_diskann_index(const std::string& name, uint32_t R, uint32_t L, float alpha) {
    if (vector_dim_ == 0) {
        throw std::runtime_error("Collection does not support vectors");
    }
    
    std::unique_lock lock(mutex_);
    
    if (indices_.find(name) != indices_.end()) {
        throw std::runtime_error("Index '" + name + "' already exists");
    }
    
    IndexCatalog& catalog = IndexCatalog::instance();
    IndexHandle handle = catalog.create_diskann_index(
        name_ + "_" + name,
        vector_dim_,
        1000000,
        R, L, alpha
    );
    
    CollectionIndexInfo info;
    info.index_id = handle.index_id();
    info.name = name;
    info.type = "diskann";
    info.status = "ready";
    info.config = {{"R", R}, {"L", L}, {"alpha", alpha}};
    
    indices_[name] = info;
    
    CALIBY_LOG_INFO("Collection", "Created DiskANN index '", name, "' on collection '", name_, "'");
}

void Collection::create_text_index(const std::string& name, const TextIndexConfig& config) {
    std::unique_lock lock(mutex_);
    
    // Check if index already exists in memory (may have been recovered on open)
    if (indices_.find(name) != indices_.end()) {
        // Index already exists - this is fine, just return silently
        // (supports idempotent create_index pattern)
        return;
    }
    
    IndexCatalog& catalog = IndexCatalog::instance();
    std::string full_name = name_ + "_" + name;
    
    IndexHandle handle;
    bool recovering = false;
    std::unique_ptr<TextIndex> text_index;
    
    // Check if index exists in catalog (recovery case)
    if (catalog.index_exists(full_name)) {
        handle = catalog.open_index(full_name);
        recovering = true;
        CALIBY_LOG_INFO("Collection", "Recovering text index '", name, "' from catalog");
        
        // Get saved metadata from catalog
        TextTypeMetadata text_config = catalog.get_text_config(full_name);
        
        CALIBY_LOG_DEBUG("Collection", "text_config: btree_slot=", text_config.btree_slot_id,
                         ", vocab=", text_config.vocab_size,
                         ", docs=", text_config.doc_count,
                         ", total_len=", text_config.total_doc_length);
        
        // Check if we have a valid BTree slot ID
        if (text_config.has_valid_btree()) {
            // Recover from persistent BTree
            text_index = TextIndex::open(
                handle.index_id(),
                text_config.btree_slot_id,
                text_config.vocab_size,
                text_config.doc_count,
                text_config.total_doc_length,
                text_config.k1,
                text_config.b
            );
            CALIBY_LOG_INFO("Collection", "Recovered text index from BTree slot ", text_config.btree_slot_id, 
                            " with ", text_config.vocab_size, " terms, ", 
                            text_config.doc_count, " docs");
        } else {
            // No valid BTree, need to rebuild from documents
            AnalyzerType analyzer_type = AnalyzerType::STANDARD;
            std::string analyzer_str(text_config.analyzer);
            if (analyzer_str == "whitespace") {
                analyzer_type = AnalyzerType::WHITESPACE;
            } else if (analyzer_str == "none") {
                analyzer_type = AnalyzerType::NONE;
            }
            
            text_index = std::make_unique<TextIndex>(
                collection_id_,
                handle.index_id(),
                analyzer_type,
                std::string(text_config.language),
                text_config.k1,
                text_config.b
            );
        }
    } else {
        // Create new index through catalog
        handle = catalog.create_text_index(
            full_name,
            config.analyzer,
            config.language,
            config.k1,
            config.b
        );
        
        // Determine analyzer type
        AnalyzerType analyzer_type = AnalyzerType::STANDARD;
        if (config.analyzer == "whitespace") {
            analyzer_type = AnalyzerType::WHITESPACE;
        } else if (config.analyzer == "none") {
            analyzer_type = AnalyzerType::NONE;
        }
        
        // Create new TextIndex object
        text_index = std::make_unique<TextIndex>(
            collection_id_,
            handle.index_id(),
            analyzer_type,
            config.language,
            config.k1,
            config.b
        );
    }
    
    // If we don't have data from recovery, index existing documents
    bool need_reindex = !recovering || text_index->doc_count() == 0;
    
    if (need_reindex) {
        uint64_t current_doc_count = doc_count_.load();
        lock.unlock();
        
        // Collect all documents for batch indexing (much faster than one-by-one)
        std::vector<uint64_t> doc_ids;
        std::vector<std::string> contents;
        doc_ids.reserve(current_doc_count);
        contents.reserve(current_doc_count);
        
        // Document IDs start at 0
        for (uint64_t doc_id = 0; doc_id < current_doc_count; ++doc_id) {
            try {
                Document doc = read_document(doc_id);
                if (!doc.content.empty()) {
                    doc_ids.push_back(doc_id);
                    contents.push_back(std::move(doc.content));
                }
            } catch (...) {
                // Document may not exist (deleted), skip
            }
        }
        
        // Callback to update doc length in id_index
        std::function<void(uint64_t, uint32_t)> update_doc_length = 
            [this](uint64_t doc_id, uint32_t doc_len) {
                this->id_index_update_doc_length(doc_id, doc_len);
            };
        
        // Batch index all documents at once
        if (!doc_ids.empty()) {
            text_index->index_batch(doc_ids, contents, &update_doc_length);
        }
        
        lock.lock();
    }
    
    // Track the index info
    CollectionIndexInfo info;
    info.index_id = handle.index_id();
    info.name = name;
    info.type = "text";
    info.status = "ready";
    info.config = {
        {"fields", config.fields},
        {"analyzer", config.analyzer},
        {"language", config.language},
        {"k1", config.k1},
        {"b", config.b}
    };
    
    indices_[name] = info;
    text_indices_[name] = std::move(text_index);
    
    // Save BTree state to catalog for persistence
    save_text_index_state(name);
    
    if (recovering) {
        CALIBY_LOG_INFO("Collection", "Recovered text index '", name, "' on collection '", name_, "'");
    } else {
        CALIBY_LOG_INFO("Collection", "Created text index '", name, "' on collection '", name_, "'");
    }
}

void Collection::create_metadata_index(const std::string& name, const MetadataIndexConfig& config) {
    std::unique_lock lock(mutex_);
    
    // Check if index already exists in memory (may have been recovered on open)
    if (indices_.find(name) != indices_.end()) {
        // Index already exists - this is fine, just return silently
        // (supports idempotent create_index pattern)
        return;
    }
    
    if (config.fields.empty()) {
        throw std::runtime_error("Metadata index requires at least one field");
    }
    
    // Verify all fields exist in schema
    for (const auto& field : config.fields) {
        if (!schema_.has_field(field)) {
            throw std::runtime_error("Field '" + field + "' not in schema");
        }
    }
    
    IndexCatalog& catalog = IndexCatalog::instance();
    std::string full_name = name_ + "_" + name;
    
    IndexHandle handle;
    bool recovering = false;
    
    // Check if index exists in catalog (recovery case)
    if (catalog.index_exists(full_name)) {
        handle = catalog.open_index(full_name);
        recovering = true;
        std::cout << "[Collection] Recovering btree index '" << name << "' from catalog" << std::endl;
    } else {
        // Create new index through catalog
        handle = catalog.create_btree_index(
            full_name,
            config.fields,
            config.unique
        );
    }
    
    // Create index info
    CollectionIndexInfo info;
    info.index_id = handle.index_id();
    info.name = name;
    info.type = "btree";
    info.status = "ready";
    
    // Store fields as JSON array
    nlohmann::json fields_json = nlohmann::json::array();
    for (const auto& field : config.fields) {
        fields_json.push_back(field);
    }
    
    info.config = {
        {"fields", fields_json},
        {"unique", config.unique}
    };
    
    // For backward compatibility, also store "field" for single-field indices
    if (config.fields.size() == 1) {
        info.config["field"] = config.fields[0];
    }
    
    indices_[name] = info;
    
    // Log index creation
    if (recovering) {
        if (config.fields.size() == 1) {
            CALIBY_LOG_INFO("Collection", "Recovered metadata index '", name, 
                           "' on field '", config.fields[0], "'");
        } else {
            std::string fields_str;
            for (size_t i = 0; i < config.fields.size(); ++i) {
                if (i > 0) fields_str += ", ";
                fields_str += config.fields[i];
            }
            CALIBY_LOG_INFO("Collection", "Recovered composite metadata index '", name, 
                           "' on fields (", fields_str, ")");
        }
    } else {
        if (config.fields.size() == 1) {
            CALIBY_LOG_INFO("Collection", "Created metadata index '", name, 
                           "' on field '", config.fields[0], "'");
        } else {
            std::string fields_str;
            for (size_t i = 0; i < config.fields.size(); ++i) {
                if (i > 0) fields_str += ", ";
                fields_str += config.fields[i];
            }
            CALIBY_LOG_INFO("Collection", "Created composite metadata index '", name, 
                           "' on fields (", fields_str, ")");
        }
    }
}

std::vector<CollectionIndexInfo> Collection::list_indices() const {
    std::shared_lock lock(mutex_);
    
    std::vector<CollectionIndexInfo> result;
    result.reserve(indices_.size());
    
    for (const auto& [name, info] : indices_) {
        result.push_back(info);
    }
    
    return result;
}

void Collection::drop_index(const std::string& name) {
    std::unique_lock lock(mutex_);
    
    auto it = indices_.find(name);
    if (it == indices_.end()) {
        throw std::runtime_error("Index '" + name + "' not found");
    }
    
    const auto& index_info = it->second;
    std::string full_name = name_ + "_" + name;
    
    // Drop from catalog for all catalog-managed indexes
    if (index_info.type == "hnsw" || index_info.type == "diskann" || 
        index_info.type == "text" || index_info.type == "btree") {
        try {
            IndexCatalog& catalog = IndexCatalog::instance();
            if (catalog.index_exists(full_name)) {
                catalog.drop_index(full_name);
            }
        } catch (const std::exception& e) {
            CALIBY_LOG_WARN("Collection", "Could not drop index from catalog: ", e.what());
        }
        
        // Remove HNSW object if present
        if (index_info.type == "hnsw") {
            hnsw_indices_.erase(name);
        }
    }
    
    // Remove text index if present
    if (index_info.type == "text") {
        text_indices_.erase(name);
    }
    
    indices_.erase(it);
    
    CALIBY_LOG_INFO("Collection", "Dropped index '", name, "'");
}

//=============================================================================
// Search Operations
//=============================================================================

std::vector<SearchResult> Collection::search_vector(
    const std::vector<float>& vector,
    const std::string& index_name,
    size_t k,
    const std::optional<FilterCondition>& where,
    const nlohmann::json& params) {
    
    if (vector_dim_ == 0) {
        throw std::runtime_error("Collection does not support vectors");
    }
    
    std::shared_lock lock(mutex_);
    
    auto it = indices_.find(index_name);
    if (it == indices_.end()) {
        throw std::runtime_error("Index '" + index_name + "' not found");
    }
    
    const auto& index_info = it->second;
    if (index_info.type != "hnsw" && index_info.type != "diskann" && index_info.type != "ivfpq") {
        throw std::runtime_error("Index '" + index_name + "' is not a vector index");
    }
    
    std::vector<SearchResult> results;
    
    // Get ef_search parameter (default to 100)
    size_t ef_search = 100;
    if (params.contains("ef_search")) {
        ef_search = params["ef_search"].get<size_t>();
    }
    
    // Handle HNSW search
    if (index_info.type == "hnsw") {
        auto hnsw_it = hnsw_indices_.find(index_name);
        if (hnsw_it == hnsw_indices_.end() || !hnsw_it->second) {
            return results;  // HNSW object not available
        }
        
        HNSWIndexBase* hnsw = hnsw_it->second.get();
        
        if (where.has_value()) {
            // Post-search filtering with over-fetching and retry
            // For 10% selectivity, need ~10x over-fetch minimum. Start higher for safety.
            // Also increase ef_search to explore more of the graph for better recall.
            const size_t max_elements = doc_count_.load();
            const std::vector<size_t> multipliers = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
            
            // Boost ef_search for filtered queries to explore more neighbors
            size_t filtered_ef_search = std::max(ef_search, static_cast<size_t>(200));
            
            for (size_t mult_idx = 0; mult_idx <= multipliers.size(); ++mult_idx) {
                size_t search_k;
                if (mult_idx < multipliers.size()) {
                    search_k = std::min(k * multipliers[mult_idx], max_elements);
                } else {
                    // Last resort: search all elements
                    search_k = max_elements;
                }
                
                auto knn_results = hnsw->searchKnn(vector.data(), search_k, filtered_ef_search);
                results.clear();
                
                for (const auto& [dist, node_id] : knn_results) {
                    uint64_t doc_id = static_cast<uint64_t>(node_id);
                    
                    try {
                        Document doc = read_document(doc_id);
                        if (!where->evaluate(doc.metadata)) {
                            continue;  // Skip non-matching documents
                        }
                        
                        SearchResult result;
                        result.doc_id = doc_id;
                        result.score = dist;
                        result.vector_score = dist;
                        result.document = std::move(doc);
                        results.push_back(std::move(result));
                        
                        if (results.size() >= k) {
                            break;
                        }
                    } catch (...) {
                        continue;  // Skip documents that can't be read
                    }
                }
                
                // If we found enough results, we're done
                if (results.size() >= k || search_k >= max_elements) {
                    break;
                }
            }
        } else {
            // No filter - simple k-NN search
            auto knn_results = hnsw->searchKnn(vector.data(), k, ef_search);
            
            for (const auto& [dist, node_id] : knn_results) {
                uint64_t doc_id = static_cast<uint64_t>(node_id);
                
                SearchResult result;
                result.doc_id = doc_id;
                result.score = dist;
                result.vector_score = dist;
                try {
                    result.document = read_document(doc_id);
                } catch (...) {
                    // Document load failed, still return result
                }
                results.push_back(std::move(result));
                
                if (results.size() >= k) {
                    break;
                }
            }
        }
    }
    
    return results;
}

std::vector<SearchResult> Collection::search_text(
    const std::string& text,
    const std::string& index_name,
    size_t k,
    const std::optional<FilterCondition>& where) {
    
    std::shared_lock lock(mutex_);
    
    auto it = indices_.find(index_name);
    if (it == indices_.end()) {
        throw std::runtime_error("Index '" + index_name + "' not found");
    }
    
    const auto& index_info = it->second;
    if (index_info.type != "text") {
        throw std::runtime_error("Index '" + index_name + "' is not a text index");
    }
    
    std::vector<SearchResult> results;
    
    // Find the text index object
    auto text_it = text_indices_.find(index_name);
    if (text_it == text_indices_.end() || !text_it->second) {
        return results;  // Text index not available
    }
    
    TextIndex* text_index = text_it->second.get();
    
    // Build doc filter if where clause is present
    std::vector<uint64_t> doc_filter_vec;
    std::vector<uint64_t>* doc_filter_ptr = nullptr;
    
    if (where.has_value()) {
        // Evaluate filter to get matching doc IDs
        // We need to release the lock temporarily to avoid deadlock
        lock.unlock();
        doc_filter_vec = evaluate_filter(*where);
        lock.lock();
        
        if (doc_filter_vec.empty()) {
            return results;  // No documents match filter
        }
        
        // Sort for binary search in text index
        std::sort(doc_filter_vec.begin(), doc_filter_vec.end());
        doc_filter_ptr = &doc_filter_vec;
    }
    
    // Callback to get doc length from id_index for BM25 scoring
    std::function<uint32_t(uint64_t)> get_doc_length = 
        [this](uint64_t doc_id) -> uint32_t {
            return this->id_index_get_doc_length(doc_id);
        };
    
    // Perform BM25 search
    auto text_results = text_index->search(text, k, doc_filter_ptr, &get_doc_length);
    
    // Convert to SearchResult format
    for (const auto& [doc_id, score] : text_results) {
        SearchResult result;
        result.doc_id = doc_id;
        result.score = score;
        result.text_score = score;
        
        // Load document
        try {
            result.document = read_document(doc_id);
        } catch (...) {
            // Document load failed, still return result
        }
        
        results.push_back(std::move(result));
    }
    
    return results;
}

std::vector<SearchResult> Collection::search_hybrid(
    const std::vector<float>& vector,
    const std::string& vector_index_name,
    const std::string& text,
    const std::string& text_index_name,
    size_t k,
    const FusionParams& fusion,
    const std::optional<FilterCondition>& where) {
    
    // Get vector search results
    auto vector_results = search_vector(vector, vector_index_name, k * 2, where);
    
    // Get text search results
    auto text_results = search_text(text, text_index_name, k * 2, where);
    
    // Fuse results
    std::unordered_map<uint64_t, SearchResult> fused;
    
    if (fusion.method == FusionMethod::RRF) {
        // Reciprocal Rank Fusion
        int rrf_k = fusion.rrf_k;
        
        for (size_t i = 0; i < vector_results.size(); ++i) {
            auto& vr = vector_results[i];
            if (fused.find(vr.doc_id) == fused.end()) {
                fused[vr.doc_id] = vr;
                fused[vr.doc_id].score = 0;
            }
            fused[vr.doc_id].score += 1.0f / (rrf_k + i + 1);
            fused[vr.doc_id].vector_score = vr.score;
        }
        
        for (size_t i = 0; i < text_results.size(); ++i) {
            auto& tr = text_results[i];
            if (fused.find(tr.doc_id) == fused.end()) {
                fused[tr.doc_id] = tr;
                fused[tr.doc_id].score = 0;
            }
            fused[tr.doc_id].score += 1.0f / (rrf_k + i + 1);
            fused[tr.doc_id].text_score = tr.score;
        }
    } else {
        // Weighted fusion
        float vec_weight = fusion.vector_weight;
        float text_weight = fusion.text_weight;
        
        // Normalize weights
        float total = vec_weight + text_weight;
        vec_weight /= total;
        text_weight /= total;
        
        // Find min/max for normalization
        float vec_min = 0, vec_max = 1, text_min = 0, text_max = 1;
        if (fusion.normalize && !vector_results.empty()) {
            vec_min = vec_max = vector_results[0].score;
            for (const auto& vr : vector_results) {
                vec_min = std::min(vec_min, vr.score);
                vec_max = std::max(vec_max, vr.score);
            }
        }
        if (fusion.normalize && !text_results.empty()) {
            text_min = text_max = text_results[0].score;
            for (const auto& tr : text_results) {
                text_min = std::min(text_min, tr.score);
                text_max = std::max(text_max, tr.score);
            }
        }
        
        auto normalize = [](float val, float min_v, float max_v) {
            if (max_v - min_v < 1e-9f) return 0.5f;
            return (val - min_v) / (max_v - min_v);
        };
        
        for (const auto& vr : vector_results) {
            if (fused.find(vr.doc_id) == fused.end()) {
                fused[vr.doc_id] = vr;
                fused[vr.doc_id].score = 0;
            }
            float norm_score = fusion.normalize ? normalize(vr.score, vec_min, vec_max) : vr.score;
            fused[vr.doc_id].score += vec_weight * norm_score;
            fused[vr.doc_id].vector_score = vr.score;
        }
        
        for (const auto& tr : text_results) {
            if (fused.find(tr.doc_id) == fused.end()) {
                fused[tr.doc_id] = tr;
                fused[tr.doc_id].score = 0;
            }
            float norm_score = fusion.normalize ? normalize(tr.score, text_min, text_max) : tr.score;
            fused[tr.doc_id].score += text_weight * norm_score;
            fused[tr.doc_id].text_score = tr.score;
        }
    }
    
    // Convert to vector and sort by score
    std::vector<SearchResult> results;
    results.reserve(fused.size());
    for (auto& [id, result] : fused) {
        results.push_back(std::move(result));
    }
    
    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });
    
    // Limit to k results
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

void Collection::flush() {
    std::unique_lock lock(mutex_);
    save_metadata();
    flush_system();
}

//=============================================================================
// Internal Methods
//=============================================================================

void Collection::load_metadata() {
    // Read metadata page (page 0 of collection file)
    // Use same PID computation as save_metadata
    PID meta_pid;
    if (bm_->supportsMultiIndexPIDs() && collection_id_ > 0) {
        meta_pid = TwoLevelPageStateArray::makePID(collection_id_, 0);
    } else {
        meta_pid = 0;
    }
    
    GuardO<CollectionMetadataPage> meta_page(meta_pid);
    
    if (!meta_page->is_valid()) {
        throw std::runtime_error("Invalid collection metadata");
    }
    
    doc_count_.store(meta_page->doc_count);
    next_doc_id_.store(meta_page->next_doc_id);
    vector_dim_ = meta_page->vector_dim;
    distance_metric_ = static_cast<DistanceMetric>(meta_page->distance_metric);
    doc_pages_head_ = meta_page->doc_pages_head;
    doc_pages_tail_ = meta_page->doc_pages_tail;
    
    // Recover DocIdIndex from persisted BTree slotId
    if (meta_page->id_index_btree_slot_id != UINT32_MAX) {
        try {
            id_index_ = std::make_unique<DocIdIndex>(meta_page->id_index_btree_slot_id);
            CALIBY_LOG_DEBUG("Collection", "Recovered DocIdIndex with BTree slotId=", 
                             meta_page->id_index_btree_slot_id);
        } catch (const std::exception& e) {
            CALIBY_LOG_WARN("Collection", "Failed to recover DocIdIndex: ", e.what());
            // Will rebuild if needed
        }
    }
    
    // Parse inline schema
    if (meta_page->inline_schema_len > 0) {
        std::string schema_json(meta_page->inline_schema, meta_page->inline_schema_len);
        schema_ = Schema::from_json(nlohmann::json::parse(schema_json));
    }
}

void Collection::save_metadata() {
    // Use page 0 for this collection's metadata (following HNSW pattern)
    // In multi-index mode, PID encodes (collection_id << 32) | local_page_id
    PID meta_pid;
    if (bm_->supportsMultiIndexPIDs() && collection_id_ > 0) {
        meta_pid = TwoLevelPageStateArray::makePID(collection_id_, 0);
        
        // Ensure allocCount is at least 1 so flushAll will iterate over page 0
        // This is necessary because GuardX doesn't update allocCount
        auto* arr = bm_->getIndexArray(collection_id_);
        if (arr) {
            u64 current = arr->allocCount.load(std::memory_order_acquire);
            if (current < 1) {
                arr->allocCount.compare_exchange_strong(current, 1);
            }
        }
    } else {
        meta_pid = 0;
    }
    
    // GuardX will read the page from disk (or create zeroed page if new)
    // and mark it dirty when we write to it
    GuardX<CollectionMetadataPage> meta_page(meta_pid);
    
    meta_page->initialize();  // Sets magic and version
    
    meta_page->doc_count = doc_count_.load();
    meta_page->next_doc_id = next_doc_id_.load();
    meta_page->vector_dim = vector_dim_;
    meta_page->distance_metric = static_cast<uint8_t>(distance_metric_);
    meta_page->doc_pages_head = doc_pages_head_;
    meta_page->doc_pages_tail = doc_pages_tail_;
    
    // Persist DocIdIndex BTree slotId for recovery
    if (id_index_) {
        meta_page->id_index_btree_slot_id = id_index_->getBTreeSlotId();
    } else {
        meta_page->id_index_btree_slot_id = UINT32_MAX;  // No index
    }
    
    // Serialize schema to inline storage
    std::string schema_json = schema_.to_json().dump();
    if (schema_json.length() < sizeof(meta_page->inline_schema)) {
        meta_page->inline_schema_len = static_cast<uint16_t>(schema_json.length());
        std::memcpy(meta_page->inline_schema, schema_json.c_str(), schema_json.length());
    }
    
    // GuardX destructor will unlock the page, keeping dirty=true
}

void Collection::save_text_index_state(const std::string& index_name) {
    auto it = text_indices_.find(index_name);
    if (it == text_indices_.end() || !it->second) {
        return;
    }
    
    TextIndex* text_index = it->second.get();
    std::string full_name = name_ + "_" + index_name;
    
    IndexCatalog& catalog = IndexCatalog::instance();
    
    // Get existing config
    TextTypeMetadata config = catalog.get_text_config(full_name);
    
    // Update with current BTree state
    config.btree_slot_id = text_index->btree_slot_id();
    config.vocab_size = text_index->vocab_size();
    config.doc_count = text_index->doc_count();
    config.total_doc_length = text_index->total_doc_length();
    
    CALIBY_LOG_DEBUG("Collection", "save_text_index_state '", index_name, 
                     "': btree_slot=", config.btree_slot_id,
                     ", vocab=", config.vocab_size,
                     ", docs=", config.doc_count,
                     ", total_len=", config.total_doc_length);
    
    // Save back to catalog
    catalog.update_text_config(full_name, config);
}

void Collection::save_all_text_index_states() {
    for (const auto& [name, text_index] : text_indices_) {
        if (text_index) {
            save_text_index_state(name);
        }
    }
}

PID Collection::allocate_page() {
    // Use buffer manager to allocate a page for this collection
    PIDAllocator* allocator = bm_->getOrCreateAllocatorForIndex(collection_id_);
    Page* page = bm_->allocPageForIndex(collection_id_, allocator);
    return bm_->toPID(page);
}

void Collection::free_page(PID page_id) {
    // TODO: Add to free list
}

void Collection::write_document(const Document& doc) {
    // Slotted page document storage with overflow support for large documents.
    // Multiple small documents are packed into a single page using slot directory.
    // Page layout:
    //   [DocumentPageHeader][SlotEntry 0][SlotEntry 1]...[SlotEntry N]
    //   [... free space ...]
    //   [Record N data][Record N-1 data]...[Record 0 data]
    //
    // Records grow from end of page backward, slots grow from start forward.
    
    // Serialize document content and metadata
    std::string serialized_meta = doc.metadata.dump();
    
    // Calculate total record size (header + content + metadata)
    size_t record_data_size = doc.content.length() + serialized_meta.length();
    size_t total_record_size = sizeof(DocumentRecordHeader) + record_data_size;
    
    // Check against absolute maximum (safety limit)
    if (record_data_size > MAX_CONTENT_SIZE + MAX_METADATA_SIZE) {
        throw std::runtime_error("Document too large: " + 
                                 std::to_string(record_data_size) + " bytes");
    }
    
    // Calculate space needed: SlotEntry + record
    size_t space_needed = sizeof(SlotEntry) + total_record_size;
    
    // Page capacity for slotted pages
    constexpr size_t page_header_size = sizeof(DocumentPageHeader);
    constexpr size_t max_record_per_page = pageSize - page_header_size - sizeof(SlotEntry);
    
    // Allocator for this collection
    PIDAllocator* allocator = bm_->getOrCreateAllocatorForIndex(collection_id_);
    
    // For very large documents, use overflow pages
    bool needs_overflow = total_record_size > max_record_per_page;
    
    // Calculate inline portion for overflow case
    constexpr size_t overflow_header_size = sizeof(OverflowPageHeader);
    constexpr size_t overflow_data_capacity = pageSize - overflow_header_size;
    
    // Prepare combined data
    std::vector<uint8_t> all_data;
    all_data.reserve(record_data_size);
    all_data.insert(all_data.end(), doc.content.begin(), doc.content.end());
    all_data.insert(all_data.end(), serialized_meta.begin(), serialized_meta.end());
    
    // Find or allocate a page with enough space
    PID target_page = 0;
    uint16_t slot_num = 0;
    
    // Try current tail page first
    if (doc_pages_tail_ != 0 && !needs_overflow) {
        GuardX<Page> page(doc_pages_tail_);
        auto* header = reinterpret_cast<DocumentPageHeader*>(page.ptr);
        
        // Check if page has enough free space
        if (header->free_space >= space_needed) {
            target_page = doc_pages_tail_;
            slot_num = header->slot_count;
            
            // Allocate slot
            auto* slots = reinterpret_cast<SlotEntry*>(
                reinterpret_cast<uint8_t*>(header) + page_header_size);
            
            // Calculate record offset (grows from end of page backward)
            uint16_t record_offset = header->free_offset - static_cast<uint16_t>(total_record_size);
            
            // Write slot entry
            slots[slot_num].offset = record_offset;
            slots[slot_num].length = static_cast<uint16_t>(total_record_size);
            slots[slot_num].flags = 0;
            slots[slot_num].reserved[0] = slots[slot_num].reserved[1] = slots[slot_num].reserved[2] = 0;
            
            // Write document record header
            auto* rec_header = reinterpret_cast<DocumentRecordHeader*>(
                reinterpret_cast<uint8_t*>(page.ptr) + record_offset);
            rec_header->doc_id = doc.id;
            rec_header->total_length = static_cast<uint32_t>(total_record_size);
            rec_header->content_length = static_cast<uint32_t>(doc.content.length());
            rec_header->metadata_length = static_cast<uint32_t>(serialized_meta.length());
            rec_header->overflow_page = 0;
            
            // Write data after record header
            std::memcpy(reinterpret_cast<uint8_t*>(rec_header + 1), all_data.data(), record_data_size);
            
            // Update page header
            header->slot_count++;
            header->free_space -= static_cast<uint16_t>(space_needed);
            header->free_offset = record_offset;
            header->dirty = true;
            
            // Update ID index
            id_index_insert(doc.id, target_page, slot_num);
            return;
        }
    }
    
    // Need to allocate a new page
    AllocGuard<Page> new_page(allocator);
    target_page = new_page.pid;
    slot_num = 0;
    
    // Initialize page header
    auto* header = reinterpret_cast<DocumentPageHeader*>(new_page.ptr);
    header->dirty = true;
    header->flags = 0;
    header->slot_count = 0;
    header->free_space = static_cast<uint16_t>(pageSize - page_header_size);
    header->free_offset = static_cast<uint16_t>(pageSize);  // Start from end
    header->next_page = 0;
    header->prev_page = doc_pages_tail_;  // Link to previous tail
    
    // Link from previous tail
    if (doc_pages_tail_ != 0) {
        GuardX<Page> prev_page(doc_pages_tail_);
        auto* prev_header = reinterpret_cast<DocumentPageHeader*>(prev_page.ptr);
        prev_header->next_page = target_page;
        prev_header->dirty = true;
    }
    
    // Update chain pointers
    if (doc_pages_head_ == 0) {
        doc_pages_head_ = target_page;
    }
    doc_pages_tail_ = target_page;
    
    // Handle overflow case for very large documents
    if (needs_overflow) {
        // For large documents, inline what fits and use overflow pages
        size_t inline_data_capacity = max_record_per_page - sizeof(DocumentRecordHeader);
        
        // Calculate record offset
        uint16_t record_offset = header->free_offset - static_cast<uint16_t>(max_record_per_page + sizeof(SlotEntry));
        
        // Allocate slot
        auto* slots = reinterpret_cast<SlotEntry*>(
            reinterpret_cast<uint8_t*>(header) + page_header_size);
        
        record_offset = static_cast<uint16_t>(pageSize - max_record_per_page);
        slots[0].offset = record_offset;
        slots[0].length = static_cast<uint16_t>(max_record_per_page);
        slots[0].flags = SlotEntry::FLAG_OVERFLOW;
        slots[0].reserved[0] = slots[0].reserved[1] = slots[0].reserved[2] = 0;
        
        // Write document record header
        auto* rec_header = reinterpret_cast<DocumentRecordHeader*>(
            reinterpret_cast<uint8_t*>(new_page.ptr) + record_offset);
        rec_header->doc_id = doc.id;
        rec_header->total_length = static_cast<uint32_t>(total_record_size);
        rec_header->content_length = static_cast<uint32_t>(doc.content.length());
        rec_header->metadata_length = static_cast<uint32_t>(serialized_meta.length());
        
        // Write inline data
        std::memcpy(reinterpret_cast<uint8_t*>(rec_header + 1), all_data.data(), inline_data_capacity);
        
        // Write overflow pages for remaining data
        size_t bytes_written = inline_data_capacity;
        PID* prev_overflow_ptr = &rec_header->overflow_page;
        
        while (bytes_written < record_data_size) {
            AllocGuard<Page> overflow_page(allocator);
            *prev_overflow_ptr = overflow_page.pid;
            
            auto* overflow_header = reinterpret_cast<OverflowPageHeader*>(overflow_page.ptr);
            overflow_header->dirty = true;
            overflow_header->parent_doc_id = doc.id;
            overflow_header->next_overflow = 0;
            
            size_t remaining = record_data_size - bytes_written;
            size_t bytes_to_write = std::min(remaining, overflow_data_capacity);
            overflow_header->continuation_length = static_cast<uint32_t>(bytes_to_write);
            
            std::memcpy(reinterpret_cast<uint8_t*>(overflow_header + 1), 
                       all_data.data() + bytes_written, bytes_to_write);
            bytes_written += bytes_to_write;
            prev_overflow_ptr = &overflow_header->next_overflow;
        }
        
        // Update page header
        header->slot_count = 1;
        header->free_space = 0;  // Page is full (large doc)
        header->free_offset = record_offset;
        
        // Update ID index
        id_index_insert(doc.id, target_page, 0);
        return;
    }
    
    // Normal case: document fits in page
    auto* slots = reinterpret_cast<SlotEntry*>(
        reinterpret_cast<uint8_t*>(header) + page_header_size);
    
    // Calculate record offset
    uint16_t record_offset = header->free_offset - static_cast<uint16_t>(total_record_size);
    
    // Write slot entry
    slots[0].offset = record_offset;
    slots[0].length = static_cast<uint16_t>(total_record_size);
    slots[0].flags = 0;
    slots[0].reserved[0] = slots[0].reserved[1] = slots[0].reserved[2] = 0;
    
    // Write document record header
    auto* rec_header = reinterpret_cast<DocumentRecordHeader*>(
        reinterpret_cast<uint8_t*>(new_page.ptr) + record_offset);
    rec_header->doc_id = doc.id;
    rec_header->total_length = static_cast<uint32_t>(total_record_size);
    rec_header->content_length = static_cast<uint32_t>(doc.content.length());
    rec_header->metadata_length = static_cast<uint32_t>(serialized_meta.length());
    rec_header->overflow_page = 0;
    
    // Write data
    std::memcpy(reinterpret_cast<uint8_t*>(rec_header + 1), all_data.data(), record_data_size);
    
    // Update page header
    header->slot_count = 1;
    header->free_space -= static_cast<uint16_t>(space_needed);
    header->free_offset = record_offset;
    
    // Update ID index
    id_index_insert(doc.id, target_page, slot_num);
}

Document Collection::read_document(uint64_t doc_id) {
    // Lookup in ID index to get (page_id, slot)
    auto location = id_index_lookup(doc_id);
    if (!location) {
        throw std::runtime_error("Document not found: " + std::to_string(doc_id));
    }
    
    PID doc_page = location->first;
    uint16_t slot_num = location->second;
    
    GuardO<Page> page(doc_page);
    
    // Read slotted page header
    constexpr size_t page_header_size = sizeof(DocumentPageHeader);
    auto* page_header = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
    
    // Validate slot number
    if (slot_num >= page_header->slot_count) {
        throw std::runtime_error("Invalid slot number for document: " + std::to_string(doc_id));
    }
    
    // Get slot entry
    auto* slots = reinterpret_cast<const SlotEntry*>(
        reinterpret_cast<const uint8_t*>(page.ptr) + page_header_size);
    const SlotEntry& slot = slots[slot_num];
    
    if (slot.is_deleted()) {
        throw std::runtime_error("Document has been deleted: " + std::to_string(doc_id));
    }
    
    // Read document record header from slot offset
    auto* header = reinterpret_cast<const DocumentRecordHeader*>(
        reinterpret_cast<const uint8_t*>(page.ptr) + slot.offset);
    
    if (header->doc_id != doc_id) {
        throw std::runtime_error("Document ID mismatch");
    }
    
    Document doc;
    doc.id = doc_id;
    
    // Calculate total data size
    size_t total_data_size = header->content_length + header->metadata_length;
    
    // Read all data (may span multiple pages for overflow docs)
    std::vector<uint8_t> all_data;
    all_data.reserve(total_data_size);
    
    // Calculate inline data capacity
    size_t inline_capacity = slot.length - sizeof(DocumentRecordHeader);
    size_t bytes_in_first = std::min(total_data_size, inline_capacity);
    
    // Read from inline portion
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(header + 1);
    all_data.insert(all_data.end(), data_ptr, data_ptr + bytes_in_first);
    
    // Read from overflow pages if needed
    PID next_overflow = header->overflow_page;
    while (next_overflow != 0 && all_data.size() < total_data_size) {
        GuardO<Page> overflow_page(next_overflow);
        auto* overflow_header = reinterpret_cast<const OverflowPageHeader*>(overflow_page.ptr);
        
        const uint8_t* overflow_data = reinterpret_cast<const uint8_t*>(overflow_header + 1);
        all_data.insert(all_data.end(), overflow_data, 
                        overflow_data + overflow_header->continuation_length);
        
        next_overflow = overflow_header->next_overflow;
    }
    
    // Extract content and metadata from combined data
    doc.content = std::string(reinterpret_cast<const char*>(all_data.data()), 
                               header->content_length);
    
    std::string meta_str(reinterpret_cast<const char*>(all_data.data() + header->content_length), 
                         header->metadata_length);
    doc.metadata = nlohmann::json::parse(meta_str);
    
    return doc;
}

void Collection::delete_document_internal(uint64_t doc_id) {
    // Lookup in ID index to get (page_id, slot)
    auto location = id_index_lookup(doc_id);
    if (!location) {
        return;  // Document not found, nothing to delete
    }
    
    PID doc_page = location->first;
    uint16_t slot_num = location->second;
    
    // Mark slot as deleted
    {
        GuardX<Page> page(doc_page);
        constexpr size_t page_header_size = sizeof(DocumentPageHeader);
        auto* page_header = reinterpret_cast<DocumentPageHeader*>(page.ptr);
        
        if (slot_num < page_header->slot_count) {
            auto* slots = reinterpret_cast<SlotEntry*>(
                reinterpret_cast<uint8_t*>(page.ptr) + page_header_size);
            slots[slot_num].flags |= SlotEntry::FLAG_DELETED;
            page_header->dirty = true;
            
            // Reclaim space (add back to free_space for potential reuse)
            page_header->free_space += slots[slot_num].length + sizeof(SlotEntry);
        }
    }
    
    // Remove from ID index
    id_index_remove(doc_id);
    
    // TODO: Free overflow pages if any
    // TODO: Compact page if fragmentation is high
}

//=============================================================================
// ID Index Methods (Persistent B-tree index)
//=============================================================================

// The ID index maps document IDs to (page_id, slot, doc_length) locations.
// Uses persistent B-tree for durability and efficient lookups.

void Collection::id_index_insert(uint64_t doc_id, PID page_id, uint16_t slot, uint32_t doc_length) {
    if (!id_index_) {
        id_index_ = std::make_unique<DocIdIndex>();
    }
    id_index_->insert(doc_id, DocIdIndex::DocLocation(page_id, slot, doc_length));
}

void Collection::id_index_update_doc_length(uint64_t doc_id, uint32_t doc_length) {
    if (!id_index_) {
        return;
    }
    auto loc = id_index_->lookup(doc_id);
    if (loc) {
        // Update with new doc_length
        id_index_->update(doc_id, DocIdIndex::DocLocation(loc->page_id, loc->slot, doc_length));
    }
}

std::optional<std::pair<PID, uint16_t>> Collection::id_index_lookup(uint64_t doc_id) {
    if (!id_index_) {
        return std::nullopt;
    }
    auto loc = id_index_->lookup(doc_id);
    if (!loc) {
        return std::nullopt;
    }
    return std::make_pair(loc->page_id, loc->slot);
}

void Collection::id_index_remove(uint64_t doc_id) {
    if (id_index_) {
        id_index_->remove(doc_id);
    }
}

uint32_t Collection::id_index_get_doc_length(uint64_t doc_id) const {
    if (!id_index_) {
        return 0;
    }
    return id_index_->get_doc_length(doc_id);
}

void Collection::rebuild_id_index() {
    // Rebuild the ID index by scanning all document pages in the chain.
    // Uses slotted page format: DocumentPageHeader followed by slot directory.
    
    uint64_t doc_count = doc_count_.load();
    if (doc_count == 0) {
        return;
    }
    
    CALIBY_LOG_INFO("Collection", "Rebuilding ID index for '", name_, 
                    "' (doc_count=", doc_count, ")");
    
    uint64_t found_docs = 0;
    constexpr size_t page_header_size = sizeof(DocumentPageHeader);
    
    // Walk the document page chain starting from doc_pages_head_
    PID current_page = doc_pages_head_;
    
    while (current_page != 0) {
        try {
            GuardO<Page> page(current_page);
            auto* page_header = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
            
            // Read all non-deleted slots
            auto* slots = reinterpret_cast<const SlotEntry*>(
                reinterpret_cast<const uint8_t*>(page.ptr) + page_header_size);
            
            for (uint16_t slot_num = 0; slot_num < page_header->slot_count; ++slot_num) {
                const SlotEntry& slot = slots[slot_num];
                
                if (slot.is_deleted()) {
                    continue;  // Skip deleted slots
                }
                
                // Read document record header
                auto* rec_header = reinterpret_cast<const DocumentRecordHeader*>(
                    reinterpret_cast<const uint8_t*>(page.ptr) + slot.offset);
                
                // Validate and add to index
                if (rec_header->doc_id > 0 || found_docs == 0) {  // doc_id 0 is valid for first doc
                    id_index_insert(rec_header->doc_id, current_page, slot_num);
                    found_docs++;
                }
            }
            
            // Move to next page in chain
            current_page = page_header->next_page;
            
        } catch (const std::exception& e) {
            CALIBY_LOG_ERROR("Collection", "Error reading page ", current_page, 
                            " during rebuild: ", e.what());
            break;
        }
    }
    
    CALIBY_LOG_INFO("Collection", "Rebuilt ID index with ", found_docs, " documents");
}

//=============================================================================
// Filter Evaluation
//=============================================================================

std::vector<uint64_t> Collection::evaluate_filter(const FilterCondition& filter) {
    std::vector<uint64_t> matching_ids;
    
    // Simple scan-based evaluation
    // TODO: Use B-tree indices for indexed fields
    
    // Scan document pages to get all doc IDs
    // (DocIdIndex doesn't support iteration, so we scan the page chain)
    constexpr size_t page_header_size = sizeof(DocumentPageHeader);
    PID current_page = doc_pages_head_;
    
    while (current_page != 0) {
        try {
            GuardO<Page> page(current_page);
            auto* page_header = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
            
            auto* slots = reinterpret_cast<const SlotEntry*>(
                reinterpret_cast<const uint8_t*>(page.ptr) + page_header_size);
            
            for (uint16_t slot_num = 0; slot_num < page_header->slot_count; ++slot_num) {
                const SlotEntry& slot = slots[slot_num];
                
                if (slot.is_deleted()) {
                    continue;
                }
                
                auto* rec_header = reinterpret_cast<const DocumentRecordHeader*>(
                    reinterpret_cast<const uint8_t*>(page.ptr) + slot.offset);
                
                uint64_t doc_id = rec_header->doc_id;
                
                // Release page guard before reading document (avoid nested locking)
                PID next_page = page_header->next_page;
                page.release();
                
                try {
                    Document doc = read_document(doc_id);
                    if (filter.evaluate(doc.metadata)) {
                        matching_ids.push_back(doc_id);
                    }
                } catch (...) {
                    // Skip documents that can't be read
                }
                
                // Re-acquire page to continue iteration
                if (slot_num + 1 < page_header->slot_count) {
                    page = GuardO<Page>(current_page);
                    page_header = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
                    slots = reinterpret_cast<const SlotEntry*>(
                        reinterpret_cast<const uint8_t*>(page.ptr) + page_header_size);
                }
            }
            
            current_page = page_header->next_page;
            
        } catch (...) {
            break;
        }
    }
    
    return matching_ids;
}

float Collection::estimate_selectivity(const FilterCondition& filter) {
    // Simple heuristic for selectivity estimation
    // TODO: Use statistics and histogram
    
    switch (filter.op) {
        case FilterOp::EQ:
            return 0.1f;  // 10% selectivity for equality
        case FilterOp::NE:
            return 0.9f;
        case FilterOp::GT:
        case FilterOp::GTE:
        case FilterOp::LT:
        case FilterOp::LTE:
            return 0.3f;  // 30% for range
        case FilterOp::IN:
            return 0.2f;
        case FilterOp::NIN:
            return 0.8f;
        case FilterOp::CONTAINS:
            return 0.1f;
        case FilterOp::AND: {
            float sel = 1.0f;
            for (const auto& child : filter.children) {
                sel *= estimate_selectivity(child);
            }
            return sel;
        }
        case FilterOp::OR: {
            float sel = 0.0f;
            for (const auto& child : filter.children) {
                sel += estimate_selectivity(child);
            }
            return std::min(sel, 1.0f);
        }
        default:
            return 0.5f;
    }
}

} // namespace caliby
