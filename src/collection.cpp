/**
 * @file collection.cpp
 * @brief Implementation of Caliby Collection System
 */

#include "collection.hpp"
#include "btree_index.hpp"
#include "text_index.hpp"
#include "array_index.hpp"
#include "catalog.hpp"
#include "hnsw.hpp"
#include "distance.hpp"
#include "logging.hpp"

#include <algorithm>
#include <ctime>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <absl/container/flat_hash_set.h>

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

    std::vector<std::pair<float, uint32_t>> searchKnnFiltered(
        const float* query, size_t k, size_t ef_search,
        const std::function<bool(uint32_t)>& filter_fn) override {
        return hnsw_.searchKnnFiltered(query, k, ef_search, filter_fn);
    }
    
    std::vector<std::pair<float, uint32_t>> searchKnnFilteredACORN(
        const float* query, size_t k, size_t ef_search,
        const std::function<bool(uint32_t)>& filter_fn,
        const std::vector<uint32_t>& matching_ids,
        float selectivity) override {
        return hnsw_.searchKnnFilteredACORN(query, k, ef_search, filter_fn, matching_ids, selectivity);
    }
    
    std::vector<std::pair<float, uint32_t>> computeDistances(
        const float* query, const std::vector<uint64_t>& candidate_ids, size_t k) override {
        return hnsw_.computeDistancesToCandidates(query, candidate_ids, k);
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

/**
 * Convert FieldType to BTreeKeyType for indexing.
 * Only scalar types can be indexed.
 */
BTreeKeyType field_type_to_btree_key_type(FieldType field_type) {
    switch (field_type) {
        case FieldType::STRING:
            return BTreeKeyType::STRING;
        case FieldType::INT:
            return BTreeKeyType::INT;
        case FieldType::FLOAT:
            return BTreeKeyType::FLOAT;
        case FieldType::BOOL:
            return BTreeKeyType::BOOL;
        case FieldType::STRING_ARRAY:
        case FieldType::INT_ARRAY:
            // Arrays are indexed as individual elements (multi-valued index)
            // For now, treat as their element type
            throw std::runtime_error("Array fields are not supported for B-tree indexing");
        default:
            throw std::runtime_error("Unknown field type for B-tree indexing");
    }
}

/**
 * Convert a MetadataValue to a BTreeKey for indexing.
 */
std::optional<BTreeKey> metadata_value_to_btree_key(const MetadataValue& value) {
    return std::visit([](auto&& v) -> std::optional<BTreeKey> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return std::nullopt;  // Null values not indexed
        } else if constexpr (std::is_same_v<T, std::string>) {
            return BTreeKey(v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return BTreeKey(v);
        } else if constexpr (std::is_same_v<T, double>) {
            return BTreeKey(v);
        } else if constexpr (std::is_same_v<T, bool>) {
            return BTreeKey(v);
        } else {
            // Arrays not supported for simple B-tree key
            return std::nullopt;
        }
    }, value);
}

/**
 * Convert a JSON value to MetadataValue.
 */
MetadataValue json_to_metadata_value(const nlohmann::json& j) {
    if (j.is_null()) {
        return std::monostate{};
    } else if (j.is_string()) {
        return j.get<std::string>();
    } else if (j.is_number_integer()) {
        return j.get<int64_t>();
    } else if (j.is_number_float()) {
        return j.get<double>();
    } else if (j.is_boolean()) {
        return j.get<bool>();
    } else if (j.is_array() && !j.empty()) {
        if (j[0].is_string()) {
            return j.get<std::vector<std::string>>();
        } else if (j[0].is_number_integer()) {
            return j.get<std::vector<int64_t>>();
        }
    }
    return std::monostate{};
}

//=============================================================================
// FilterCondition Implementation
//=============================================================================

FilterCondition FilterCondition::from_json(const nlohmann::json& j) {
    FilterCondition cond;
    
    // Handle verbose format: {"field": "name", "op": "eq", "value": ...}
    if (j.contains("field") && j.contains("op") && j.contains("value")) {
        cond.field = j["field"].get<std::string>();
        std::string op_str = j["op"].get<std::string>();
        
        if (op_str == "eq" || op_str == "$eq") cond.op = FilterOp::EQ;
        else if (op_str == "ne" || op_str == "$ne") cond.op = FilterOp::NE;
        else if (op_str == "gt" || op_str == "$gt") cond.op = FilterOp::GT;
        else if (op_str == "gte" || op_str == "$gte") cond.op = FilterOp::GTE;
        else if (op_str == "lt" || op_str == "$lt") cond.op = FilterOp::LT;
        else if (op_str == "lte" || op_str == "$lte") cond.op = FilterOp::LTE;
        else if (op_str == "in" || op_str == "$in") cond.op = FilterOp::IN;
        else if (op_str == "nin" || op_str == "$nin") cond.op = FilterOp::NIN;
        else if (op_str == "contains" || op_str == "$contains") cond.op = FilterOp::CONTAINS;
        else throw std::runtime_error("Unknown filter operator: " + op_str);
        
        const auto& val = j["value"];
        if (val.is_string()) {
            cond.value = val.get<std::string>();
        } else if (val.is_number_integer()) {
            cond.value = val.get<int64_t>();
        } else if (val.is_number_float()) {
            cond.value = val.get<double>();
        } else if (val.is_boolean()) {
            cond.value = val.get<bool>();
        } else if (val.is_array()) {
            if (!val.empty() && val[0].is_string()) {
                cond.value = val.get<std::vector<std::string>>();
            } else {
                cond.value = val.get<std::vector<int64_t>>();
            }
        }
        return cond;
    }
    
    // Handle verbose "and"/"or" format (without $)
    if (j.contains("and")) {
        cond.op = FilterOp::AND;
        for (const auto& child : j["and"]) {
            cond.children.push_back(from_json(child));
        }
        return cond;
    }
    
    if (j.contains("or")) {
        cond.op = FilterOp::OR;
        for (const auto& child : j["or"]) {
            cond.children.push_back(from_json(child));
        }
        return cond;
    }
    
    if (j.contains("not")) {
        // Handle NOT as a special case: negate the inner condition
        // We'll implement NOT as a single-element NOT (AND with negation)
        cond.op = FilterOp::NOT;
        cond.children.push_back(from_json(j["not"]));
        return cond;
    }
    
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
    
    if (j.contains("$not")) {
        cond.op = FilterOp::NOT;
        cond.children.push_back(from_json(j["$not"]));
        return cond;
    }
    
    // Single field condition
    for (auto& [key, val] : j.items()) {
        cond.field = key;
        
        if (val.is_object()) {
            // Operator form: {"field": {"$gt": 10}} or {"field": {"$gt": 10, "$lt": 100}}
            // If multiple operators, create AND condition
            std::vector<FilterCondition> sub_conditions;
            
            for (auto& [op_str, op_val] : val.items()) {
                FilterCondition sub_cond;
                sub_cond.field = key;
                
                if (op_str == "$eq") sub_cond.op = FilterOp::EQ;
                else if (op_str == "$ne") sub_cond.op = FilterOp::NE;
                else if (op_str == "$gt") sub_cond.op = FilterOp::GT;
                else if (op_str == "$gte") sub_cond.op = FilterOp::GTE;
                else if (op_str == "$lt") sub_cond.op = FilterOp::LT;
                else if (op_str == "$lte") sub_cond.op = FilterOp::LTE;
                else if (op_str == "$in") sub_cond.op = FilterOp::IN;
                else if (op_str == "$nin") sub_cond.op = FilterOp::NIN;
                else if (op_str == "$contains") sub_cond.op = FilterOp::CONTAINS;
                else throw std::runtime_error("Unknown filter operator: " + op_str);
                
                // Convert JSON value to MetadataValue
                if (op_val.is_string()) {
                    sub_cond.value = op_val.get<std::string>();
                } else if (op_val.is_number_integer()) {
                    sub_cond.value = op_val.get<int64_t>();
                } else if (op_val.is_number_float()) {
                    sub_cond.value = op_val.get<double>();
                } else if (op_val.is_boolean()) {
                    sub_cond.value = op_val.get<bool>();
                } else if (op_val.is_array()) {
                    // Check first element type
                    if (!op_val.empty() && op_val[0].is_string()) {
                        sub_cond.value = op_val.get<std::vector<std::string>>();
                    } else {
                        sub_cond.value = op_val.get<std::vector<int64_t>>();
                    }
                }
                sub_conditions.push_back(std::move(sub_cond));
            }
            
            // If single operator, use directly; otherwise create AND
            if (sub_conditions.size() == 1) {
                cond = std::move(sub_conditions[0]);
            } else if (sub_conditions.size() > 1) {
                cond.op = FilterOp::AND;
                cond.field.clear();  // AND conditions don't have a field
                cond.children = std::move(sub_conditions);
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
        case FilterOp::NOT: {
            if (children.empty()) return true;
            return !children[0].evaluate(metadata);
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
                    10000000,                 // max_elements (10M)
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
                            Document doc = collection->read_document(doc_id, false);  // Only need content for text indexing
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
            
            // Update btree indices for this document
            update_btree_indices_for_document(doc.id, doc.metadata, false);
            
            // Update array indices for this document
            update_array_indices_for_document(doc.id, doc.metadata, false);
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
        nlohmann::json old_metadata = doc.metadata;
        
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
        
        // Remove old metadata from btree indices
        update_btree_indices_for_document(ids[i], old_metadata, true);
        // Remove old metadata from array indices
        update_array_indices_for_document(ids[i], old_metadata, true);
        
        // Delete old and write new
        delete_document_internal(ids[i]);
        write_document(doc);
        
        // Add new metadata to btree indices
        update_btree_indices_for_document(ids[i], doc.metadata, false);
        // Add new metadata to array indices
        update_array_indices_for_document(ids[i], doc.metadata, false);
    }
}

void Collection::delete_docs(const std::vector<uint64_t>& ids) {
    std::unique_lock lock(mutex_);
    
    for (uint64_t id : ids) {
        // Read document to get metadata for btree index removal
        try {
            Document doc = read_document(id);
            update_btree_indices_for_document(id, doc.metadata, true);
            update_array_indices_for_document(id, doc.metadata, true);
        } catch (...) {
            // Document may not exist - continue with deletion
        }
        
        delete_document_internal(id);
        doc_count_.fetch_sub(1);
    }
    
    save_metadata();
}

size_t Collection::delete_docs(const FilterCondition& where) {
    std::unique_lock lock(mutex_);
    
    std::vector<uint64_t> matching_ids = evaluate_filter(where);
    
    for (uint64_t id : matching_ids) {
        // Read document to get metadata for btree index removal
        try {
            Document doc = read_document(id);
            update_btree_indices_for_document(id, doc.metadata, true);
            update_array_indices_for_document(id, doc.metadata, true);
        } catch (...) {
            // Document may not exist - continue with deletion
        }
        
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
            10000000,  // max_elements (10M) - will grow as needed
            M,
            ef_construction
        );
    }
    
    // Create actual HNSW object using collection's distance metric (will recover if data exists)
    auto hnsw = createHNSWIndex(
        distance_metric_,  // Use collection's distance metric
        10000000,       // max_elements (10M)
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
                Document doc = read_document(doc_id, false);  // Just checking existence, no metadata needed
                
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
                Document doc = read_document(doc_id, false);  // Only need content for text indexing
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
    
    // Create in-memory BTreeMetadataIndex for single-field indices
    // (Multi-field composite indices require different handling)
    if (config.fields.size() == 1) {
        const std::string& field_name = config.fields[0];
        const FieldDef* field_def = schema_.get_field(field_name);
        if (field_def) {
            try {
                BTreeKeyType key_type = field_type_to_btree_key_type(field_def->type);
                auto btree_idx = std::make_unique<BTreeMetadataIndex>(
                    field_name, key_type, config.unique);
                
                // Always populate index with existing documents
                // (The in-memory BTreeMetadataIndex needs to be rebuilt from documents)
                if (doc_count_.load() > 0) {
                    // Scan existing documents and index them
                    constexpr size_t page_header_size = sizeof(DocumentPageHeader);
                    PID current_page = doc_pages_head_;
                    
                    while (current_page != 0) {
                        try {
                            GuardO<Page> page(current_page);
                            auto* page_header = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
                            auto* slots = reinterpret_cast<const SlotEntry*>(
                                reinterpret_cast<const uint8_t*>(page.ptr) + page_header_size);
                            
                            PID next_page = page_header->next_page;
                            uint16_t slot_count = page_header->slot_count;
                            
                            for (uint16_t slot_num = 0; slot_num < slot_count; ++slot_num) {
                                const SlotEntry& slot = slots[slot_num];
                                if (slot.is_deleted()) continue;
                                
                                auto* rec_header = reinterpret_cast<const DocumentRecordHeader*>(
                                    reinterpret_cast<const uint8_t*>(page.ptr) + slot.offset);
                                uint64_t doc_id = rec_header->doc_id;
                                
                                // Release page guard before reading doc
                                page.release();
                                
                                try {
                                    Document doc = read_document(doc_id);
                                    if (doc.metadata.contains(field_name)) {
                                        MetadataValue mv = json_to_metadata_value(doc.metadata[field_name]);
                                        auto btree_key = metadata_value_to_btree_key(mv);
                                        if (btree_key) {
                                            btree_idx->insert(*btree_key, doc_id);
                                        }
                                    }
                                } catch (...) {
                                    // Skip documents that can't be read
                                }
                                
                                // Re-acquire page if needed
                                if (slot_num + 1 < slot_count) {
                                    page = GuardO<Page>(current_page);
                                    page_header = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
                                    slots = reinterpret_cast<const SlotEntry*>(
                                        reinterpret_cast<const uint8_t*>(page.ptr) + page_header_size);
                                }
                            }
                            
                            current_page = next_page;
                        } catch (...) {
                            break;
                        }
                    }
                }
                
                btree_indices_[name] = std::move(btree_idx);
                CALIBY_LOG_INFO("Collection", "Created in-memory btree index '", name, 
                               "' for field '", field_name, "', total btree_indices_.size()=", 
                               btree_indices_.size());
            } catch (const std::exception& e) {
                CALIBY_LOG_WARN("Collection", "Could not create in-memory btree index for '", 
                              name, "': ", e.what());
            }
        }
    }
    
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
            // Selectivity-based adaptive strategy for filtered search
            // 
            // 1. Estimate selectivity using histograms
            // 2. Choose strategy based on selectivity:
            //    - HIGH selectivity (>10%): Pre-filter + brute force (most docs match, cheaper to scan)
            //    - MEDIUM selectivity (0.5%-10%): Pre-filter + ACORN filtered HNSW
            //    - LOW selectivity (≤0.5%): Pre-filter + brute-force for perfect recall
            //
            // Note: Post-filtering is ONLY used when no index exists for the filter field,
            // because for indexed fields, pre-filtering is always more efficient.
            
            ef_search = 300;  // Higher ef_search for filtered searches
            float selectivity = estimate_selectivity(*where);
            
            // Configurable thresholds - lower thresholds to use more brute force for better recall
            float high_selectivity_threshold = 0.15f;  // 15% - increased for more ACORN usage
            float low_selectivity_threshold = 0.01f;  // 1% - increased for more brute force at low selectivity
            if (params.contains("high_selectivity_threshold")) {
                high_selectivity_threshold = params["high_selectivity_threshold"].get<float>();
            }
            if (params.contains("low_selectivity_threshold")) {
                low_selectivity_threshold = params["low_selectivity_threshold"].get<float>();
            }
            
            // Debug logging for selectivity estimation
            CALIBY_LOG_INFO("Collection", "Filtered search: estimated selectivity=", selectivity * 100, 
                           "% threshold_high=", high_selectivity_threshold * 100, 
                           "% threshold_low=", low_selectivity_threshold * 100, "%");
            
            // Always try to pre-filter first - it's more efficient than post-filter
            // Post-filter is only a fallback when evaluate_filter returns empty AND no index exists
            std::vector<uint64_t> matching_ids = evaluate_filter(*where);
            
            // Check if filter uses an indexed field - if so, empty results means no matches
            // (not "couldn't evaluate"), so we should return empty, not fall back to post-filter
            bool filter_uses_index = false;
            if (!where->field.empty()) {
                if (where->op == FilterOp::CONTAINS && find_array_index_for_field(where->field)) {
                    filter_uses_index = true;
                } else if (find_btree_index_for_field(where->field)) {
                    filter_uses_index = true;
                }
            }
            
            if (!matching_ids.empty()) {
                // Pre-filtering succeeded - use appropriate search strategy
                bool use_brute_force = false;
                if (selectivity > high_selectivity_threshold) {
                    // HIGH selectivity: many docs match, use ACORN filtered SEARCH with a large candidate set for best performance
                    use_brute_force = false;
                }
                else if (selectivity <= low_selectivity_threshold) {
                    // HIGH or LOW selectivity: brute-force for best performance/recall
                    // - HIGH: many docs match, brute-force is efficient
                    // - LOW: few docs match, brute-force gives perfect recall
                    use_brute_force = true;
                } else {
                    // MEDIUM selectivity: use filtered HNSW for scalability
                    // Increase threshold for better recall - brute force is more reliable for moderate-sized candidate sets
                    size_t brute_force_threshold = 5000;  // Increased from 1000 for better recall
                    if (params.contains("filter_bruteforce_threshold")) {
                        brute_force_threshold = params["filter_bruteforce_threshold"].get<size_t>();
                    }
                    use_brute_force = (matching_ids.size() <= brute_force_threshold);
                }
                
                CALIBY_LOG_INFO("Collection", "Pre-filter: matching_ids=", matching_ids.size(), 
                               " use_brute_force=", use_brute_force);
                
                if (use_brute_force) {
                    // BRUTE-FORCE: Compute distances to all candidates
                    auto prefilter_results = hnsw->computeDistances(vector.data(), matching_ids, k);
                    
                    for (const auto& [dist, node_id] : prefilter_results) {
                        uint64_t doc_id = static_cast<uint64_t>(node_id);
                        
                        SearchResult result;
                        result.doc_id = doc_id;
                        result.score = dist;
                        result.vector_score = dist;
                        try {
                            result.document = read_document(doc_id, true);  // Include metadata for search results
                        } catch (...) {}
                        results.push_back(std::move(result));
                        
                        if (results.size() >= k) break;
                    }
                } else {
                    // ACORN-STYLE FILTERED HNSW
                    absl::flat_hash_set<uint32_t> allowed_ids;
                    allowed_ids.reserve(matching_ids.size());
                    std::vector<uint32_t> matching_ids_u32;
                    matching_ids_u32.reserve(matching_ids.size());
                    for (uint64_t id : matching_ids) {
                        allowed_ids.insert(static_cast<uint32_t>(id));
                        matching_ids_u32.push_back(static_cast<uint32_t>(id));
                    }
                    
                    // Debug: Verify matching_ids_u32 is valid before ACORN call
                    // if (matching_ids_u32.size() > 10000) {
                    //     CALIBY_LOG_WARN("Collection", "ACORN pre-check: matching_ids_u32.size()=", matching_ids_u32.size(),
                    //                    " [0]=", matching_ids_u32[0], " [5000]=", matching_ids_u32[5000],
                    //                    " [10000]=", matching_ids_u32[10000],
                    //                    " capacity=", matching_ids_u32.capacity());
                    // }

                    auto filtered_results = hnsw->searchKnnFilteredACORN(
                        vector.data(), k, ef_search,
                        [&allowed_ids](uint32_t node_id) {
                            return allowed_ids.contains(node_id);
                        },
                        matching_ids_u32,
                        selectivity);

                    for (const auto& [dist, node_id] : filtered_results) {
                        uint64_t doc_id = static_cast<uint64_t>(node_id);
                        
                        SearchResult result;
                        result.doc_id = doc_id;
                        result.score = dist;
                        result.vector_score = dist;
                        try {
                            result.document = read_document(doc_id, true);  // Include metadata for search results
                        } catch (...) {}
                        results.push_back(std::move(result));
                        
                        if (results.size() >= k) break;
                    }
                }
            } else if (filter_uses_index) {
                // Filter used an index but returned empty - no documents match
                // Return empty results immediately (don't fall back to post-filter)
                CALIBY_LOG_INFO("Collection", "Pre-filter returned empty for indexed field - no matches");
                // results is already empty, just return it
            } else {
                // FALLBACK: Post-filtering when no pre-filter index exists
                // This is slow but necessary for unindexed fields
                CALIBY_LOG_WARN("Collection", "Post-filter fallback: no matching IDs from pre-filter");
                
                size_t postfilter_max_k = std::max(k * 100, static_cast<size_t>(1000));
                if (params.contains("postfilter_max_k")) {
                    postfilter_max_k = params["postfilter_max_k"].get<size_t>();
                }
                double postfilter_growth_factor = 1.5;
                if (params.contains("postfilter_growth_factor")) {
                    postfilter_growth_factor = params["postfilter_growth_factor"].get<double>();
                }
                if (postfilter_growth_factor < 1.1) {
                    postfilter_growth_factor = 1.5;
                }

                size_t current_k = std::max(static_cast<size_t>(1), k);
                bool satisfied = false;
                
                while (!satisfied && current_k <= postfilter_max_k) {
                    auto knn_results = hnsw->searchKnn(vector.data(), current_k, ef_search);

                    results.clear();
                    for (const auto& [dist, node_id] : knn_results) {
                        uint64_t doc_id = static_cast<uint64_t>(node_id);

                        try {
                            Document doc = read_document(doc_id);
                            if (!where->evaluate(doc.metadata)) {
                                continue;
                            }

                            SearchResult result;
                            result.doc_id = doc_id;
                            result.score = dist;
                            result.vector_score = dist;
                            result.document = std::move(doc);
                            results.push_back(std::move(result));

                            if (results.size() >= k) {
                                satisfied = true;
                                break;
                            }
                        } catch (...) {}
                    }

                    if (satisfied) break;
                    
                    size_t next_k = static_cast<size_t>(current_k * postfilter_growth_factor);
                    if (next_k <= current_k) next_k = current_k + 1;
                    current_k = std::min(postfilter_max_k, next_k);
                    // Searched with the max k already: stop instead of looping forever.
                    if (current_k >= postfilter_max_k) break;
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
                    result.document = read_document(doc_id, true);  // Include metadata for search results
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
            result.document = read_document(doc_id, true);  // Include metadata for search results
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
            markPageDirty(header);
            
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
    markPageDirty(header);
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
        markPageDirty(prev_header);
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
            markPageDirty(overflow_header);
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

Document Collection::read_document(uint64_t doc_id, bool parse_metadata) {
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
    
    // Calculate inline data capacity
    size_t inline_capacity = slot.length - sizeof(DocumentRecordHeader);
    
    // Fast path: all data is inline (no overflow pages)
    // Avoid intermediate vector allocation by copying directly to doc.content
    if (header->overflow_page == 0 && total_data_size <= inline_capacity) {
        const char* data_ptr = reinterpret_cast<const char*>(header + 1);
        
        // Copy content directly
        doc.content.assign(data_ptr, header->content_length);
        
        // Parse metadata if needed
        if (parse_metadata && header->metadata_length > 0) {
            std::string_view meta_str(data_ptr + header->content_length, header->metadata_length);
            doc.metadata = nlohmann::json::parse(meta_str);
        }
        
        return doc;
    }
    
    // Slow path: data spans overflow pages - need to collect all data first
    std::vector<uint8_t> all_data;
    all_data.reserve(total_data_size);
    
    size_t bytes_in_first = std::min(total_data_size, inline_capacity);
    
    // Read from inline portion
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(header + 1);
    all_data.insert(all_data.end(), data_ptr, data_ptr + bytes_in_first);
    
    // Read from overflow pages
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
    
    if (parse_metadata) {
        std::string meta_str(reinterpret_cast<const char*>(all_data.data() + header->content_length), 
                             header->metadata_length);
        doc.metadata = nlohmann::json::parse(meta_str);
    }
    
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
            markPageDirty(page_header);
            
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

// Helper to convert filter value to BTreeKey
static std::optional<BTreeKey> filter_value_to_btree_key(const FilterCondition& filter) {
    return std::visit([](auto&& v) -> std::optional<BTreeKey> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, std::string>) {
            return BTreeKey(v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return BTreeKey(v);
        } else if constexpr (std::is_same_v<T, double>) {
            return BTreeKey(v);
        } else if constexpr (std::is_same_v<T, bool>) {
            return BTreeKey(v);
        } else {
            return std::nullopt;  // Arrays not supported
        }
    }, filter.value);
}

std::vector<uint64_t> Collection::evaluate_filter(const FilterCondition& filter) {
    // Debug: Log btree index lookup attempt
    CALIBY_LOG_DEBUG("Collection", "evaluate_filter: field='", filter.field, 
                    "', op=", static_cast<int>(filter.op), 
                    ", btree_indices_.size()=", btree_indices_.size(),
                    ", array_indices_.size()=", array_indices_.size());
    
    // Try to use array index for CONTAINS operation on indexed array fields
    if (!filter.field.empty() && filter.op == FilterOp::CONTAINS) {
        ArrayIndex* array_idx = find_array_index_for_field(filter.field);
        
        CALIBY_LOG_DEBUG("Collection", "evaluate_filter: find_array_index_for_field('", 
                        filter.field, "') returned ", (array_idx ? "non-null" : "null"));
        
        if (array_idx) {
            // Extract the value to search for
            return std::visit([array_idx](auto&& v) -> std::vector<uint64_t> {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    return array_idx->lookup(v);
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    return array_idx->lookup(v);
                } else {
                    return {};
                }
            }, filter.value);
        }
    }
    
    // Try to use B-tree index for simple conditions on indexed fields
    if (!filter.field.empty()) {
        BTreeMetadataIndex* btree_idx = find_btree_index_for_field(filter.field);
        
        CALIBY_LOG_DEBUG("Collection", "evaluate_filter: find_btree_index_for_field('", 
                        filter.field, "') returned ", (btree_idx ? "non-null" : "null"));
        
        if (btree_idx) {
            auto btree_key = filter_value_to_btree_key(filter);
            
            if (btree_key) {
                switch (filter.op) {
                    case FilterOp::EQ: {
                        // Exact lookup using B-tree
                        return btree_idx->lookup(*btree_key);
                    }
                    case FilterOp::GT: {
                        // Greater than (exclusive)
                        return btree_idx->greater_than(*btree_key, false);
                    }
                    case FilterOp::GTE: {
                        // Greater than or equal (inclusive)
                        return btree_idx->greater_than(*btree_key, true);
                    }
                    case FilterOp::LT: {
                        // Less than (exclusive)
                        return btree_idx->less_than(*btree_key, false);
                    }
                    case FilterOp::LTE: {
                        // Less than or equal (inclusive)
                        return btree_idx->less_than(*btree_key, true);
                    }
                    default:
                        // NE, IN, NIN - fall through to scan
                        break;
                }
            }
        }
    }
    
    // Handle AND/OR with btree/array index optimization where possible
    if (filter.op == FilterOp::AND && !filter.children.empty()) {
        // OPTIMIZATION: Check for range query pattern (GT/GTE + LT/LTE on same field)
        // This is much more efficient than two separate scans + intersection
        
        // Recursively flatten nested ANDs to find all range conditions
        std::unordered_map<std::string, std::vector<const FilterCondition*>> field_conditions;
        std::vector<const FilterCondition*> other_conditions_flat;
        
        std::function<void(const FilterCondition&)> collect_conditions;
        collect_conditions = [&](const FilterCondition& cond) {
            if (cond.op == FilterOp::AND) {
                // Flatten nested AND
                for (const auto& child : cond.children) {
                    collect_conditions(child);
                }
            } else if (!cond.field.empty() && 
                       (cond.op == FilterOp::GT || cond.op == FilterOp::GTE ||
                        cond.op == FilterOp::LT || cond.op == FilterOp::LTE)) {
                // Range condition on a field
                field_conditions[cond.field].push_back(&cond);
            } else {
                // Other condition
                other_conditions_flat.push_back(&cond);
            }
        };
        
        for (const auto& child : filter.children) {
            collect_conditions(child);
        }
        
        // Check each field for range query optimization opportunity
        for (auto& [field_name, conditions] : field_conditions) {
            if (conditions.size() >= 2) {
                // Look for GT/GTE and LT/LTE pair
                const FilterCondition* lower = nullptr;
                const FilterCondition* upper = nullptr;
                bool lower_inclusive = false;
                bool upper_inclusive = false;
                
                for (const auto* cond : conditions) {
                    if (cond->op == FilterOp::GT) { lower = cond; lower_inclusive = false; }
                    else if (cond->op == FilterOp::GTE) { lower = cond; lower_inclusive = true; }
                    else if (cond->op == FilterOp::LT) { upper = cond; upper_inclusive = false; }
                    else if (cond->op == FilterOp::LTE) { upper = cond; upper_inclusive = true; }
                }
                
                if (lower && upper) {
                    // Found range query pattern - use btree range_scan
                    BTreeMetadataIndex* btree_idx = find_btree_index_for_field(field_name);
                    if (btree_idx) {
                        auto lower_key = filter_value_to_btree_key(*lower);
                        auto upper_key = filter_value_to_btree_key(*upper);
                        
                        if (lower_key && upper_key) {
                            // Use efficient range scan
                            auto range_iter = btree_idx->range_scan(
                                lower_key, upper_key, 
                                lower_inclusive, upper_inclusive);
                            
                            // Collect all results from range iterator
                            std::vector<uint64_t> result = range_iter.collect();
                            
                            // Filter by any remaining conditions not part of the range
                            std::vector<const FilterCondition*> other_conditions;
                            for (const auto& child : filter.children) {
                                if (&child != lower && &child != upper) {
                                    other_conditions.push_back(&child);
                                }
                            }
                            
                            if (!other_conditions.empty()) {
                                std::vector<uint64_t> final_result;
                                for (uint64_t doc_id : result) {
                                    try {
                                        Document doc = read_document(doc_id);
                                        bool passes = true;
                                        for (const auto* cond : other_conditions) {
                                            if (!cond->evaluate(doc.metadata)) {
                                                passes = false;
                                                break;
                                            }
                                        }
                                        if (passes) {
                                            final_result.push_back(doc_id);
                                        }
                                    } catch (...) {
                                        // Skip documents that can't be read
                                    }
                                }
                                return final_result;
                            }
                            return result;
                        }
                    }
                }
            }
        }
        
        // Fall back to original AND handling
        // For AND, we can use btree/array index for any indexed child, then filter the rest
        // Find the most selective indexed condition
        std::vector<uint64_t> result;
        bool have_index_result = false;
        std::vector<const FilterCondition*> remaining_conditions;
        
        for (const auto& child : filter.children) {
            if (!child.field.empty()) {
                // Check for array index on CONTAINS operations first
                if (child.op == FilterOp::CONTAINS) {
                    ArrayIndex* array_idx = find_array_index_for_field(child.field);
                    if (array_idx) {
                        if (!have_index_result) {
                            result = evaluate_filter(child);
                            have_index_result = true;
                        } else {
                            auto other = evaluate_filter(child);
                            std::sort(result.begin(), result.end());
                            std::sort(other.begin(), other.end());
                            std::vector<uint64_t> intersection;
                            std::set_intersection(result.begin(), result.end(),
                                                other.begin(), other.end(),
                                                std::back_inserter(intersection));
                            result = std::move(intersection);
                        }
                        continue;
                    }
                }
                
                // Check for btree index on scalar operations
                BTreeMetadataIndex* btree_idx = find_btree_index_for_field(child.field);
                auto btree_key = filter_value_to_btree_key(child);
                
                if (btree_idx && btree_key && 
                    (child.op == FilterOp::EQ || child.op == FilterOp::GT || 
                     child.op == FilterOp::GTE || child.op == FilterOp::LT || 
                     child.op == FilterOp::LTE)) {
                    
                    if (!have_index_result) {
                        // Use first btree-indexed condition as base
                        result = evaluate_filter(child);
                        have_index_result = true;
                    } else {
                        // Intersect with additional btree results
                        auto other = evaluate_filter(child);
                        std::sort(result.begin(), result.end());
                        std::sort(other.begin(), other.end());
                        std::vector<uint64_t> intersection;
                        std::set_intersection(result.begin(), result.end(),
                                            other.begin(), other.end(),
                                            std::back_inserter(intersection));
                        result = std::move(intersection);
                    }
                    continue;
                }
            }
            remaining_conditions.push_back(&child);
        }
        
        if (have_index_result) {
            // Filter remaining conditions by scanning matched docs
            if (!remaining_conditions.empty()) {
                std::vector<uint64_t> final_result;
                for (uint64_t doc_id : result) {
                    try {
                        Document doc = read_document(doc_id);
                        bool passes = true;
                        for (const auto* cond : remaining_conditions) {
                            if (!cond->evaluate(doc.metadata)) {
                                passes = false;
                                break;
                            }
                        }
                        if (passes) {
                            final_result.push_back(doc_id);
                        }
                    } catch (...) {
                        // Skip documents that can't be read
                    }
                }
                return final_result;
            }
            return result;
        }
    }
    
    if (filter.op == FilterOp::OR && !filter.children.empty()) {
        // For OR, collect results from all children and union
        std::vector<uint64_t> result;
        for (const auto& child : filter.children) {
            auto child_result = evaluate_filter(child);
            result.insert(result.end(), child_result.begin(), child_result.end());
        }
        // Remove duplicates
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
    
    if (filter.op == FilterOp::NOT && !filter.children.empty()) {
        // For NOT, get all doc IDs and subtract the inner condition's matches
        auto inner_matches = evaluate_filter(filter.children[0]);
        std::unordered_set<uint64_t> inner_set(inner_matches.begin(), inner_matches.end());
        
        // Scan all documents and collect those NOT in inner_matches
        std::vector<uint64_t> result;
        constexpr size_t page_hdr_size = sizeof(DocumentPageHeader);
        PID cur_page = doc_pages_head_;
        
        while (cur_page != 0) {
            try {
                GuardO<Page> page(cur_page);
                auto* page_hdr = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
                auto* slots = reinterpret_cast<const SlotEntry*>(
                    reinterpret_cast<const uint8_t*>(page.ptr) + page_hdr_size);
                
                for (uint16_t slot_num = 0; slot_num < page_hdr->slot_count; ++slot_num) {
                    const SlotEntry& slot = slots[slot_num];
                    if (slot.is_deleted()) continue;
                    
                    auto* header = reinterpret_cast<const DocumentRecordHeader*>(
                        reinterpret_cast<const uint8_t*>(page.ptr) + slot.offset);
                    uint64_t doc_id = header->doc_id;
                    
                    // Include if NOT in inner matches
                    if (inner_set.find(doc_id) == inner_set.end()) {
                        result.push_back(doc_id);
                    }
                }
                cur_page = page_hdr->next_page;
            } catch (...) {
                break;
            }
        }
        return result;
    }
    
    // Fall back to scan-based evaluation
    std::vector<uint64_t> matching_ids;
    
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

uint64_t Collection::estimate_cardinality(const FilterCondition& filter) const {
    // Use histogram-based cardinality estimation for better accuracy
    uint64_t total_docs = doc_count_.load();
    if (total_docs == 0) return 0;
    
    switch (filter.op) {
        case FilterOp::EQ: {
            // Try to find btree index for this field
            BTreeMetadataIndex* btree = find_btree_index_for_field(filter.field);
            if (btree) {
                auto btree_key = metadata_value_to_btree_key(filter.value);
                if (btree_key) {
                    return btree->histogram().estimate_eq(*btree_key);
                }
            }
            // Conservative fallback: assume 1% selectivity
            // This ensures we use pre-filtering for better recall
            return total_docs / 100;
        }
        
        case FilterOp::NE: {
            // NOT equal - complement of EQ
            BTreeMetadataIndex* btree = find_btree_index_for_field(filter.field);
            if (btree) {
                auto btree_key = metadata_value_to_btree_key(filter.value);
                if (btree_key) {
                    uint64_t eq_count = btree->histogram().estimate_eq(*btree_key);
                    return total_docs > eq_count ? total_docs - eq_count : 0;
                }
            }
            // Fallback: assume 90% selectivity
            return total_docs * 9 / 10;
        }
        
        case FilterOp::GT:
        case FilterOp::GTE: {
            BTreeMetadataIndex* btree = find_btree_index_for_field(filter.field);
            if (btree) {
                auto btree_key = metadata_value_to_btree_key(filter.value);
                if (btree_key) {
                    bool inclusive = (filter.op == FilterOp::GTE);
                    return btree->histogram().estimate_range(btree_key, std::nullopt, inclusive, true);
                }
            }
            // Fallback: assume 30% selectivity
            return total_docs * 3 / 10;
        }
        
        case FilterOp::LT:
        case FilterOp::LTE: {
            BTreeMetadataIndex* btree = find_btree_index_for_field(filter.field);
            if (btree) {
                auto btree_key = metadata_value_to_btree_key(filter.value);
                if (btree_key) {
                    bool inclusive = (filter.op == FilterOp::LTE);
                    return btree->histogram().estimate_range(std::nullopt, btree_key, true, inclusive);
                }
            }
            // Fallback: assume 30% selectivity
            return total_docs * 3 / 10;
        }
        
        case FilterOp::IN: {
            // IN (v1, v2, ...) - sum of equality estimates
            // Value can be std::vector<std::string> or std::vector<int64_t>
            BTreeMetadataIndex* btree = find_btree_index_for_field(filter.field);
            if (btree) {
                std::vector<BTreeKey> keys;
                
                if (std::holds_alternative<std::vector<std::string>>(filter.value)) {
                    const auto& values = std::get<std::vector<std::string>>(filter.value);
                    keys.reserve(values.size());
                    for (const auto& v : values) {
                        keys.push_back(BTreeKey(v));
                    }
                } else if (std::holds_alternative<std::vector<int64_t>>(filter.value)) {
                    const auto& values = std::get<std::vector<int64_t>>(filter.value);
                    keys.reserve(values.size());
                    for (int64_t v : values) {
                        keys.push_back(BTreeKey(v));
                    }
                }
                
                if (!keys.empty()) {
                    return btree->histogram().estimate_in(keys);
                }
            }
            // Fallback: assume 20% selectivity
            return total_docs / 5;
        }
        
        case FilterOp::NIN: {
            // NOT IN - complement of IN
            BTreeMetadataIndex* btree = find_btree_index_for_field(filter.field);
            if (btree) {
                std::vector<BTreeKey> keys;
                
                if (std::holds_alternative<std::vector<std::string>>(filter.value)) {
                    const auto& values = std::get<std::vector<std::string>>(filter.value);
                    keys.reserve(values.size());
                    for (const auto& v : values) {
                        keys.push_back(BTreeKey(v));
                    }
                } else if (std::holds_alternative<std::vector<int64_t>>(filter.value)) {
                    const auto& values = std::get<std::vector<int64_t>>(filter.value);
                    keys.reserve(values.size());
                    for (int64_t v : values) {
                        keys.push_back(BTreeKey(v));
                    }
                }
                
                if (!keys.empty()) {
                    uint64_t in_count = btree->histogram().estimate_in(keys);
                    return total_docs > in_count ? total_docs - in_count : 0;
                }
            }
            // Fallback: assume 80% selectivity
            return total_docs * 4 / 5;
        }
        
        case FilterOp::CONTAINS: {
            // Array contains - try to use array index for accurate estimate
            ArrayIndex* arr_idx = find_array_index_for_field(filter.field);
            if (arr_idx) {
                // Try to look up the actual posting list size
                try {
                    std::vector<uint64_t> matches;
                    if (std::holds_alternative<std::string>(filter.value)) {
                        matches = arr_idx->lookup(std::get<std::string>(filter.value));
                    } else if (std::holds_alternative<int64_t>(filter.value)) {
                        matches = arr_idx->lookup(std::get<int64_t>(filter.value));
                    }
                    // Return actual count from posting list
                    return matches.size();
                } catch (...) {
                    // Fallback on error
                }
            }
            // Conservative fallback: assume 1% selectivity (most array element matches are sparse)
            // This ensures we use pre-filtering instead of expensive post-filtering
            return total_docs / 100;
        }
        
        case FilterOp::AND: {
            // Assume independence and multiply selectivities
            // First, try to detect bounded range queries on the same field
            // e.g., {$and: [{field: {$gt: X}}, {field: {$lt: Y}}]}
            // This is common and should be estimated using histogram.estimate_range(X, Y)
            
            // Group conditions by field (recursively flatten nested ANDs)
            std::unordered_map<std::string, std::vector<const FilterCondition*>> field_conditions;
            std::vector<const FilterCondition*> other_conditions;
            
            std::function<void(const FilterCondition&)> collect_range_conditions;
            collect_range_conditions = [&](const FilterCondition& cond) {
                if (cond.op == FilterOp::AND) {
                    for (const auto& child : cond.children) {
                        collect_range_conditions(child);
                    }
                } else if (cond.op == FilterOp::GT || cond.op == FilterOp::GTE ||
                           cond.op == FilterOp::LT || cond.op == FilterOp::LTE) {
                    field_conditions[cond.field].push_back(&cond);
                } else {
                    other_conditions.push_back(&cond);
                }
            };
            
            for (const auto& child : filter.children) {
                collect_range_conditions(child);
            }
            
            uint64_t estimate = total_docs;
            std::unordered_set<const FilterCondition*> handled;
            
            // Handle bounded range queries efficiently
            for (const auto& [field, conds] : field_conditions) {
                const FilterCondition* lower = nullptr;
                const FilterCondition* upper = nullptr;
                bool lower_inclusive = false;
                bool upper_inclusive = false;
                
                for (const FilterCondition* c : conds) {
                    if (c->op == FilterOp::GT || c->op == FilterOp::GTE) {
                        if (!lower) {
                            lower = c;
                            lower_inclusive = (c->op == FilterOp::GTE);
                        }
                    } else if (c->op == FilterOp::LT || c->op == FilterOp::LTE) {
                        if (!upper) {
                            upper = c;
                            upper_inclusive = (c->op == FilterOp::LTE);
                        }
                    }
                }
                
                // If we have both lower and upper bound, use bounded range estimate
                if (lower && upper) {
                    BTreeMetadataIndex* btree = find_btree_index_for_field(field);
                    if (btree) {
                        auto lower_key = metadata_value_to_btree_key(lower->value);
                        auto upper_key = metadata_value_to_btree_key(upper->value);
                        if (lower_key && upper_key) {
                            uint64_t range_card = btree->histogram().estimate_range(
                                lower_key, upper_key, lower_inclusive, upper_inclusive);
                            double sel = static_cast<double>(range_card) / total_docs;
                            estimate = static_cast<uint64_t>(estimate * sel);
                            handled.insert(lower);
                            handled.insert(upper);
                        }
                    } else {
                        // No btree index - use conservative estimate (5% for bounded range)
                        estimate = static_cast<uint64_t>(estimate * 0.05);
                        handled.insert(lower);
                        handled.insert(upper);
                    }
                }
            }
            
            // Process remaining range conditions not part of bounded ranges
            for (const auto& [field, conds] : field_conditions) {
                for (const FilterCondition* c : conds) {
                    if (handled.count(c)) continue;
                    uint64_t child_card = estimate_cardinality(*c);
                    double sel = static_cast<double>(child_card) / total_docs;
                    estimate = static_cast<uint64_t>(estimate * sel);
                    handled.insert(c);
                }
            }
            
            // Process other (non-range) conditions
            for (const FilterCondition* c : other_conditions) {
                uint64_t child_card = estimate_cardinality(*c);
                double sel = static_cast<double>(child_card) / total_docs;
                estimate = static_cast<uint64_t>(estimate * sel);
            }
            
            return std::max(estimate, static_cast<uint64_t>(1));
        }
        
        case FilterOp::OR: {
            // Use inclusion-exclusion principle (simplified)
            // For simplicity, use max(children) + partial sum heuristic
            uint64_t max_card = 0;
            uint64_t sum_card = 0;
            for (const auto& child : filter.children) {
                uint64_t child_card = estimate_cardinality(child);
                max_card = std::max(max_card, child_card);
                sum_card += child_card;
            }
            // Heuristic: average of max and sum, capped at total
            return std::min((max_card + sum_card) / 2, total_docs);
        }
        
        case FilterOp::NOT: {
            // Estimate NOT as total - inner cardinality
            if (filter.children.empty()) return total_docs;
            uint64_t child_card = estimate_cardinality(filter.children[0]);
            return (child_card >= total_docs) ? 0 : total_docs - child_card;
        }
        
        default:
            return total_docs / 2;
    }
}

float Collection::estimate_selectivity(const FilterCondition& filter) {
    // Use histogram-based cardinality estimation
    uint64_t total_docs = doc_count_.load();
    if (total_docs == 0) return 0.0f;
    
    uint64_t cardinality = estimate_cardinality(filter);
    return static_cast<float>(cardinality) / static_cast<float>(total_docs);
}

BTreeMetadataIndex* Collection::find_btree_index_for_field(const std::string& field_name) const {
    // Search through all btree indices to find one that indexes the given field
    for (const auto& [index_name, btree_idx] : btree_indices_) {
        if (btree_idx && btree_idx->field_name() == field_name) {
            return btree_idx.get();
        }
    }
    return nullptr;
}

void Collection::update_btree_indices_for_document(uint64_t doc_id, const nlohmann::json& metadata, bool is_delete) {
    // Update all btree indices with this document's metadata
    for (const auto& [index_name, btree_idx] : btree_indices_) {
        if (!btree_idx) continue;
        
        const std::string& field_name = btree_idx->field_name();
        if (!metadata.contains(field_name)) continue;
        
        try {
            MetadataValue mv = json_to_metadata_value(metadata.at(field_name));
            auto btree_key = metadata_value_to_btree_key(mv);
            if (!btree_key) continue;
            
            if (is_delete) {
                btree_idx->remove(*btree_key, doc_id);
            } else {
                btree_idx->insert(*btree_key, doc_id);
            }
        } catch (const std::exception& e) {
            // Log but don't fail - btree index is an optimization
            CALIBY_LOG_WARN("Collection", "Failed to update btree index '", index_name, 
                          "' for doc ", doc_id, ": ", e.what());
        }
    }
}

ArrayIndex* Collection::find_array_index_for_field(const std::string& field_name) const {
    // Search through all array indices to find one that indexes the given field
    auto it = array_indices_.find(field_name);
    if (it != array_indices_.end() && it->second) {
        return it->second.get();
    }
    return nullptr;
}

void Collection::update_array_indices_for_document(uint64_t doc_id, const nlohmann::json& metadata, bool is_delete) {
    // Update all array indices with this document's metadata
    for (const auto& [field_name, array_idx] : array_indices_) {
        if (!array_idx) continue;
        
        try {
            if (is_delete) {
                // Mark document as deleted in this array index
                if (metadata.contains(field_name)) {
                    const auto& field_value = metadata.at(field_name);
                    if (field_value.is_array()) {
                        array_idx->remove_document(doc_id);
                    }
                }
            } else {
                // Always clear tombstone first, regardless of whether this index contains this doc
                // (the document may have been deleted and is now being re-added)
                array_idx->undelete_document(doc_id);
                
                // Then index if the metadata contains this field
                if (metadata.contains(field_name)) {
                    const auto& field_value = metadata.at(field_name);
                    if (!field_value.is_array()) continue;
                    
                    // Extract elements based on type
                    if (array_idx->element_type() == ArrayElementType::STRING) {
                        std::vector<std::string> elements;
                        for (const auto& elem : field_value) {
                            if (elem.is_string()) {
                                elements.push_back(elem.get<std::string>());
                            }
                        }
                        if (!elements.empty()) {
                            array_idx->index_document(doc_id, elements);
                        }
                    } else if (array_idx->element_type() == ArrayElementType::INT) {
                        std::vector<int64_t> elements;
                        for (const auto& elem : field_value) {
                            if (elem.is_number_integer()) {
                                elements.push_back(elem.get<int64_t>());
                            }
                        }
                        if (!elements.empty()) {
                            array_idx->index_document(doc_id, elements);
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            CALIBY_LOG_WARN("Collection", "Failed to update array index for field '", field_name, 
                          "' doc ", doc_id, ": ", e.what());
        }
    }
}

void Collection::create_array_index(const std::string& name, const std::string& field_name) {
    std::unique_lock lock(mutex_);
    
    // Check if index already exists
    if (indices_.find(name) != indices_.end()) {
        throw std::runtime_error("Index '" + name + "' already exists");
    }
    
    // Verify field exists and is an array type
    const FieldDef* field_def = schema_.get_field(field_name);
    if (!field_def) {
        throw std::runtime_error("Field '" + field_name + "' not found in schema");
    }
    
    ArrayElementType element_type;
    if (field_def->type == FieldType::STRING_ARRAY) {
        element_type = ArrayElementType::STRING;
    } else if (field_def->type == FieldType::INT_ARRAY) {
        element_type = ArrayElementType::INT;
    } else {
        throw std::runtime_error("Field '" + field_name + "' is not an array type (STRING_ARRAY or INT_ARRAY)");
    }
    
    // Create full index name for catalog
    std::string full_name = name_ + "_" + name;
    
    // Create through catalog
    IndexCatalog& catalog = IndexCatalog::instance();
    IndexHandle handle;
    bool recovering = false;
    
    if (catalog.index_exists(full_name)) {
        handle = catalog.open_index(full_name);
        recovering = true;
        CALIBY_LOG_INFO("Collection", "Recovering array index '", name, "' for field '", field_name, "'");
    } else {
        // Create new index entry in catalog
        // For array index, we use a special type "array" 
        handle = catalog.create_btree_index(full_name, {field_name}, false);
    }
    
    // Create index info
    CollectionIndexInfo info;
    info.index_id = handle.index_id();
    info.name = name;
    info.type = "array";
    info.status = "ready";
    info.config = {
        {"field", field_name},
        {"element_type", element_type == ArrayElementType::STRING ? "string" : "int"}
    };
    
    indices_[name] = info;
    
    // Create the actual ArrayIndex object
    try {
        auto array_idx = std::make_unique<ArrayIndex>(
            collection_id_, handle.index_id(), field_name, element_type);
        
        // Populate index with existing documents
        // Release the lock before iterating through documents to avoid deadlock
        uint64_t docs_to_index = doc_count_.load();
        if (docs_to_index > 0 && !recovering) {
            CALIBY_LOG_INFO("Collection", "Populating array index '", name, "' with existing documents...");
            
            // Release the lock for document reading
            lock.unlock();
            
            uint64_t indexed_count = 0;
            
            // We need to iterate through all documents - use the document pages
            constexpr size_t page_header_size = sizeof(DocumentPageHeader);
            PID current_page = doc_pages_head_;
            
            while (current_page != 0) {
                PID next_page = 0;
                std::vector<uint64_t> doc_ids_to_index;
                
                // First pass: collect all doc_ids from this page
                {
                    GuardO<Page> page(current_page);
                    auto* page_header = reinterpret_cast<const DocumentPageHeader*>(page.ptr);
                    auto* slots = reinterpret_cast<const SlotEntry*>(
                        reinterpret_cast<const uint8_t*>(page.ptr) + page_header_size);
                    
                    next_page = page_header->next_page;
                    uint16_t slot_count = page_header->slot_count;
                    
                    for (uint16_t slot_num = 0; slot_num < slot_count; ++slot_num) {
                        const SlotEntry& slot = slots[slot_num];
                        if (slot.is_deleted()) continue;
                        
                        auto* rec_header = reinterpret_cast<const DocumentRecordHeader*>(
                            reinterpret_cast<const uint8_t*>(page.ptr) + slot.offset);
                        doc_ids_to_index.push_back(rec_header->doc_id);
                    }
                }  // Page guard released here
                
                // Second pass: index each document (no page guard held)
                for (uint64_t doc_id : doc_ids_to_index) {
                    try {
                        Document doc = read_document(doc_id);
                        if (doc.metadata.contains(field_name)) {
                            const auto& field_value = doc.metadata[field_name];
                            if (field_value.is_array()) {
                                if (element_type == ArrayElementType::STRING) {
                                    std::vector<std::string> elements;
                                    for (const auto& elem : field_value) {
                                        if (elem.is_string()) {
                                            elements.push_back(elem.get<std::string>());
                                        }
                                    }
                                    if (!elements.empty()) {
                                        array_idx->index_document(doc_id, elements);
                                        indexed_count++;
                                    }
                                } else {
                                    std::vector<int64_t> elements;
                                    for (const auto& elem : field_value) {
                                        if (elem.is_number_integer()) {
                                            elements.push_back(elem.get<int64_t>());
                                        }
                                    }
                                    if (!elements.empty()) {
                                        array_idx->index_document(doc_id, elements);
                                        indexed_count++;
                                    }
                                }
                            }
                        }
                    } catch (...) {
                        // Skip documents that can't be read
                    }
                }
                
                current_page = next_page;
            }
            
            CALIBY_LOG_INFO("Collection", "Indexed ", indexed_count, " documents in array index '", name, "'");
            
            // Re-acquire lock before modifying array_indices_
            lock.lock();
        }
        
        array_indices_[field_name] = std::move(array_idx);
        CALIBY_LOG_INFO("Collection", "Created array index '", name, "' for field '", field_name, "'");
        
    } catch (const std::exception& e) {
        CALIBY_LOG_WARN("Collection", "Could not create array index '", name, "': ", e.what());
        throw;
    }
}

} // namespace caliby
