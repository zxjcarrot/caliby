#include <pybind11/functional.h>  // Needed for py::function
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "calico.hpp"
#include "catalog.hpp"
#include "collection.hpp"
#include "diskann.hpp"
#include "distance.hpp"
#include "hnsw.hpp"
#include "ivfpq.hpp"
#include "text_index.hpp"
#include "logging.hpp"

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

size_t getCurrentRSS() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}

#elif defined(__APPLE__)
#include <mach/mach.h>

size_t getCurrentRSS() {
    mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
}

#else  // Assuming Linux
#include <unistd.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <string>

size_t getCurrentRSS() {
    std::ifstream status_file("/proc/self/status");
    if (!status_file) {
        return 0;
    }
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            try {
                // Format is "VmRSS:     12345 kB"
                size_t rss_kb = std::stoul(line.substr(line.find_first_of("0123456789")));
                return rss_kb * 1024;  // Convert kB to bytes
            } catch (...) {
                return 0;
            }
        }
    }
    return 0;
}
#endif

namespace py = pybind11;

// --- Forward declare init/shutdown functions from calico.cpp ---
void initialize_system();
void shutdown_system();
void flush_system();
void set_buffer_config(float virtgb, float physgb);

// --- Automatic Initializer ---
// NOTE: Auto-initialization is disabled to allow set_buffer_config() to be called first.
// The system will be initialized lazily on first index creation.
// struct SystemInitializer {
//     SystemInitializer() {
//         std::cout << "Calico module loading: Initializing system..." << std::endl;
//         initialize_system();
//         std::cout << "Calico system initialized." << std::endl;
//     }
//     ~SystemInitializer() {
//         std::cout << "Calico module unloading: Shutting down system..." << std::endl;
//         shutdown_system();
//     }
// };
// static SystemInitializer initializer;

// --- Define the specific HNSW instance we will bind ---
// For this binding, we will use a fixed dimension of 128 to match the explicit
// template instantiation in hnsw.cpp. This is a common approach for high-performance bindings.
constexpr size_t HNSW_DIM = 128;
using HnswIndexType = HNSW<hnsw_distance::SIMDAcceleratedL2>;

// --- Python Module Definition ---
PYBIND11_MODULE(caliby, m) {
    m.doc() = "Python bindings for the Calico Index(B-Tree, HNSW)";
    m.attr("__version__") = "0.1.0.dev20260129183920";
    
    // Register cleanup function to be called at module unload
    auto cleanup = []() {
        CALIBY_LOG_INFO("Bindings", "Calico module unloading: Shutting down system...");
        shutdown_system();
    };
    m.add_object("_cleanup", py::capsule(cleanup));

    // This line is the key. It registers a translator that catches
    // standard C++ exceptions (like std::runtime_error, std::invalid_argument)
    // and converts them into corresponding Python exceptions (RuntimeError, ValueError).
    // The error message from C++ will be preserved.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception& e) {
            // py::exception is a Pybind11 helper to create a Python exception object.
            // PyExc_RuntimeError is the Python C-API equivalent for RuntimeError.
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // py::class_<HnswIndexType>(m, "HnswIndex")
    //     .def(py::init<u64, size_t, size_t, bool>(), py::arg("max_elements"), py::arg("M") = 16,
    //          py::arg("ef_construction") = 200, py::arg("enable_prefetch") = true,
    //          "Initializes a new, empty HNSW index with runtime parameters.")
    m.def("flush_storage", []() { flush_system(); },
          "Flushes all dirty pages managed by the Calico buffer pool to persistent storage.");
    
    m.def("open", [](const std::string& data_dir, bool cleanup_if_exist) {
        // Reset system closed flag - system is being (re)opened
        system_closed = false;
        
        // Initialize the buffer manager and catalog with the specified directory
        initialize_system();
        
        // If cleanup_if_exist, also clear the simple IndexCatalog to prevent stale data
        // from being used when creating new indexes with explicit index_ids
        if (cleanup_if_exist && bm_ptr && bm_ptr->indexCatalog) {
            bm_ptr->indexCatalog->clear();
        }
        
        // Connect catalog to buffer manager for proper index registration
        // Use &bm to get address of the BufferManager (bm is defined as *bm_ptr)
        caliby::IndexCatalog::instance().setBufferManager(&bm);
        caliby::IndexCatalog::instance().initialize(data_dir, cleanup_if_exist);
    }, py::arg("data_dir"), py::arg("cleanup_if_exist") = false,
    "Open caliby with a specific data directory for storing index files and catalog. "
    "This must be called before creating or opening any indexes. "
    "Only one process can open a data directory at a time (enforced via file lock). "
    "If the directory contains existing indexes, proper recovery will be performed. "
    "\n\nParameters:\n"
    "  data_dir: Path to the data directory\n"
    "  cleanup_if_exist: If True, removes all existing indexes and data in the directory (default: False)");
    
    m.def("close", []() {
        // Mark system as closed BEFORE unregistering index arrays
        // This prevents Collection destructors from trying to flush after close
        system_closed = true;
        
        // Flush all changes
        flush_system();
        
        // Persist index translation array capacities before shutdown
        // This ensures proper recovery of array sizes on restart
        if (bm_ptr != nullptr) {
            bm_ptr->persistIndexCapacities();
        }
        
        // Shutdown catalog
        caliby::IndexCatalog::instance().shutdown();
        
        // Unregister all non-zero indexes from the BufferManager
        // This allows open() to be called again with fresh state
        if (bm_ptr != nullptr) {
            bm_ptr->unregisterAllNonZero();
        }
    },
    "Close caliby, flushing all changes and releasing the data directory lock. "
    "Should be called before program exit for clean shutdown.");
    
    m.def("set_buffer_config", [](float size_gb, float virtgb) {
        if (bm_ptr != nullptr) {
            throw std::runtime_error("Cannot change buffer config after system is initialized. "
                                     "Call set_buffer_config() before creating any indexes.");
        }
        set_buffer_config(virtgb, size_gb);
    }, py::arg("size_gb"), py::arg("virtgb") = 24.0f,
    "Configure buffer pool sizes (in GB) before system initialization. "
    "Must be called before creating any indexes. "
    "size_gb: physical buffer size (resident pages), "
    "virtgb: virtual buffer size (total pages, defaults to 24GB - auto-computed per-index).");
    
    // --- Logging Configuration ---
    py::enum_<caliby::LogLevel>(m, "LogLevel")
        .value("DEBUG", caliby::LogLevel::DEBUG)
        .value("INFO", caliby::LogLevel::INFO)
        .value("WARN", caliby::LogLevel::WARN)
        .value("ERROR", caliby::LogLevel::ERROR)
        .value("OFF", caliby::LogLevel::OFF)
        .export_values();
    
    m.def("set_log_level", [](caliby::LogLevel level) {
        caliby::set_log_level(level);
    }, py::arg("level"),
    "Set the logging level using LogLevel enum. "
    "Levels: DEBUG (verbose), INFO (normal), WARN (warnings only), ERROR (errors only), OFF (silent).");
    
    m.def("set_log_level", [](const std::string& level) {
        caliby::set_log_level(caliby::string_to_log_level(level));
    }, py::arg("level"),
    "Set the logging level using a string. "
    "Valid values: 'DEBUG', 'INFO', 'WARN', 'ERROR', 'OFF' (case-insensitive).");
    
    m.def("get_log_level", []() {
        return caliby::get_log_level();
    }, "Get the current logging level.");
    
    m.def("force_evict_buffer_portion", [](float portion) {
        if (bm_ptr) {
            bm_ptr->forceEvictPortion(portion);
        } else {
            throw std::runtime_error("BufferManager not initialized");
        }
    }, py::arg("portion") = 0.5,
    "Force eviction of a portion of the buffer pool (for testing hole punching). "
    "Portion must be between 0.0 and 1.0 (default 0.5 = 50%).");

        py::class_<HnswIndexType>(m, "HnswIndex")
                .def(py::init([](u64 max_elements, size_t dim, size_t M, size_t ef_construction,
                                bool enable_prefetch, bool skip_recovery, uint32_t index_id,
                                const std::string& name) {
                    // Ensure system is initialized before creating index
                    initialize_system();
                    
                    uint32_t final_index_id = index_id;
                    std::string final_name = name;
                    
                    // If index_id is 0 and catalog is initialized, create a catalog entry
                    caliby::IndexCatalog& catalog = caliby::IndexCatalog::instance();
                    if (final_index_id == 0 && catalog.is_initialized()) {
                        // Generate a name if not provided
                        if (final_name.empty()) {
                            final_name = "hnsw_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
                        }
                        
                        // Create catalog entry which sets up per-index file
                        caliby::IndexHandle handle = catalog.create_hnsw_index(
                            final_name, 
                            static_cast<uint32_t>(dim),
                            max_elements,
                            M,
                            ef_construction
                        );
                        final_index_id = handle.index_id();
                    }
                    
                    return new HnswIndexType(max_elements, dim, M, ef_construction,
                                           enable_prefetch, skip_recovery, final_index_id, final_name);
                }),
                         py::arg("max_elements"),
                         py::arg("dim") = HNSW_DIM,
                         py::arg("M") = 16,
                         py::arg("ef_construction") = 200,
                         py::arg("enable_prefetch") = true,
                         py::arg("skip_recovery") = false,
                         py::arg("index_id") = 0,
                         py::arg("name") = "",
                         "Initializes a new, empty HNSW index with runtime parameters. Set skip_recovery to True to rebuild the index from scratch. index_id is used for multi-index isolation. name is an optional identifier for the index. "
                         "If caliby.open() was called and index_id is 0, the index will be automatically registered with the catalog.")
        .def(
            "flush",
            [](HnswIndexType&) { flush_system(); },
            "Flushes all dirty pages managed by Calico to persistent storage.")
        .def("was_recovered", &HnswIndexType::wasRecovered,
             "Returns True if the underlying index state was recovered from existing storage.")
        .def("get_name", &HnswIndexType::getName,
             "Returns the name of the index.")

        .def(
            "add_points",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> items,
               size_t num_threads = 0) {
                // Check input array dimensions
                if (items.ndim() != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional (n_samples, n_features)");
                }
                const size_t dim = self.getDim();
                if (items.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " + std::to_string(items.shape(1)));
                }

                // Use the parallel implementation
                std::span<const float> data_span(items.data(), items.size());
                self.addPoint_parallel(data_span, num_threads);
            },
            py::arg("items"), py::arg("num_threads") = 0,
            R"doc(
Adds a batch of points from a NumPy array to the index in parallel.

Parameters:
-----------
items : numpy.ndarray
    A 2D NumPy array of shape (num_items, dim) containing the vectors to be added.
num_threads : int, optional
    The number of threads to use for adding points. If 0, it defaults to the
    number of hardware cores. (default: 0)
)doc")

        .def(
            "search_knn",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, size_t k,
               size_t ef_search_param, bool stats = false) -> py::object {
                // Check input query vector dimensions
                if (query.ndim() != 1) {
                    throw std::runtime_error("Query must be a 1-dimensional NumPy array");
                }
                const size_t dim = self.getDim();
                if (query.shape(0) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vector has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " + std::to_string(query.shape(0)));
                }

                const float* query_ptr = query.data();
                std::vector<std::pair<float, u32>> result_vec;
                if (stats) {
                    result_vec = self.searchKnn<true>(query_ptr, k, ef_search_param);
                } else {
                    result_vec = self.searchKnn<false>(query_ptr, k, ef_search_param);
                }

                // Create two Python lists to hold labels and distances
                py::list labels;
                py::list distances;

                for (const auto& pair : result_vec) {
                    distances.append(pair.first);  // distance
                    labels.append(pair.second);    // label (node_id)
                }

                // Return a tuple of two NumPy arrays, similar to hnswlib
                return py::make_tuple(py::array(labels), py::array(distances));
            },
            py::arg("query"), py::arg("k"), py::arg("ef_search"), py::arg("stats") = false,
            "Searches for the k-nearest neighbors for a given query vector. Returns (labels, distances).")

        .def(
            "search_knn_parallel",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
               size_t ef_search_param, size_t num_threads = 0, bool stats = false) -> py::object {
                // --- 1. Validate Input ---
                if (queries.ndim() != 2) {
                    throw std::runtime_error("Queries must be a 2-dimensional NumPy array of shape (num_queries, dim)");
                }
                const size_t dim = self.getDim();
                if (queries.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vectors have incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(queries.shape(1)));
                }
                if (k == 0) {
                    throw std::runtime_error("k must be > 0");
                }

                size_t num_queries = queries.shape(0);
                if (num_queries == 0) {
                    // Explicitly create shape vectors to resolve ambiguity.
                    // We use ssize_t for dimensions, as is common in pybind11/NumPy.
                    std::vector<ssize_t> shape = {0, static_cast<ssize_t>(k)};
                    py::array_t<int64_t> empty_labels(shape);  // Use int64_t for labels
                    py::array_t<float> empty_distances(shape);
                    return py::make_tuple(empty_labels, empty_distances);
                }

                // --- 2. Prepare C++ data and call the parallel search function ---
                std::span<const float> query_span(queries.data(), queries.size());
                std::vector<std::vector<std::pair<float, u32>>> all_results_vec;
                if (stats) {
                    all_results_vec = self.searchKnn_parallel<true>(query_span, k, ef_search_param, num_threads);
                } else {
                    all_results_vec = self.searchKnn_parallel(query_span, k, ef_search_param, num_threads);
                }

                // --- 3. Create output NumPy arrays ---
                // Use int64_t for labels to be consistent with hnswlib and allow -1 as a sentinel
                // value.
                std::vector<ssize_t> result_shape = {static_cast<ssize_t>(num_queries), static_cast<ssize_t>(k)};
                py::array_t<int64_t> labels_arr(result_shape);
                py::array_t<float> distances_arr(result_shape);

                auto labels_ptr = labels_arr.mutable_data();
                auto distances_ptr = distances_arr.mutable_data();

                // --- 4. Populate the output arrays from the C++ result vector ---
                for (size_t i = 0; i < num_queries; ++i) {
                    const auto& results_for_one_query = all_results_vec[i];
                    for (size_t j = 0; j < k; ++j) {
                        if (j < results_for_one_query.size()) {
                            distances_ptr[i * k + j] = results_for_one_query[j].first;
                            labels_ptr[i * k + j] = static_cast<int64_t>(results_for_one_query[j].second);
                        } else {
                            // Fill remaining spots if fewer than k results were returned.
                            // This matches hnswlib's behavior.
                            distances_ptr[i * k + j] = std::numeric_limits<float>::infinity();
                            labels_ptr[i * k + j] = -1;  // Use -1 as the invalid label marker
                        }
                    }
                }

                // --- 5. Return a tuple of the two NumPy arrays ---
                return py::make_tuple(labels_arr, distances_arr);
            },
            py::arg("queries"), py::arg("k"), py::arg("ef_search"), py::arg("num_threads") = 0,
            py::arg("stats") = false,
            R"doc(
    Searches for the k-nearest neighbors for a batch of query vectors in parallel.

    Parameters:
    -----------
    queries : numpy.ndarray
        A 2D NumPy array of shape (num_queries, dim) containing the query vectors.
    k : int
        The number of nearest neighbors to search for.
    
    ef_search_param: int
        The ef_search num.

    num_threads : int, optional
        The number of threads to use for the search. If 0, it defaults to the
        number of hardware cores. (default: 0).
    
    stats: bool, optional

    Returns:
    --------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing two 2D NumPy arrays:
        - labels: An array of shape (num_queries, k) and dtype int64 with the IDs of the nearest neighbors.
        - distances: An array of shape (num_queries, k) and dtype float32 with the corresponding distances.
    )doc")

        .def(
            "add_items",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> data,
               size_t num_threads = 0) {
                // --- 1. Validate Input ---
                if (data.ndim() != 2) {
                    throw std::runtime_error("Data must be a 2-dimensional NumPy array of shape (num_items, dim)");
                }
                const size_t dim = self.getDim();
                if (data.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Data vectors have incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " + std::to_string(data.shape(1)));
                }

                if (data.shape(0) == 0) {
                    return;  // Nothing to add
                }

                // --- 2. Call the C++ parallel implementation ---
                std::span<const float> data_span(data.data(), data.size());
                self.addPoint_parallel(data_span, num_threads);
            },
            py::arg("data"), py::arg("num_threads") = 0,
            R"doc(
    Adds a batch of items to the index in parallel.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D NumPy array of shape (num_items, dim) containing the vectors to be added.
    num_threads : int, optional
        The number of threads to use for adding items. If 0, it defaults to the
        number of hardware cores. (default: 0)
    )doc")

        .def("get_buffer_manager_alloc_count", &HnswIndexType::getBufferManagerAllocCount,
             "Returns the current allocation count from the buffer manager.")

        .def(
            "get_index_info", &HnswIndexType::getIndexInfo,
            "Returns comprehensive information about the HNSW index including configuration, metadata, and statistics.")

        .def("get_dim", &HnswIndexType::getDim, "Returns the dimensionality of vectors managed by the index.")

        .def("reset_stats", &HnswIndexType::resetStats,
             "Resets live statistics counters like distance computations and search hops.")

        .def(
            "get_stats_string", [](HnswIndexType& self) { return self.getStats().toString(); },
            "Returns a formatted string of the current performance and graph structure statistics.")

        .def(
            "get_stats",
            [](HnswIndexType& self) -> py::dict {
                HNSWStats stats = self.getStats();
                py::dict result;

                result["dist_comps"] = stats.dist_comps.load();
                result["search_hops"] = stats.search_hops.load();
                result["num_levels"] = stats.num_levels;
                result["nodes_per_level"] = py::cast(stats.nodes_per_level);
                result["links_per_level"] = py::cast(stats.links_per_level);
                result["avg_neighbors_per_level"] = py::cast(stats.avg_neighbors_per_level);
                result["avg_neighbors_total"] = stats.avg_neighbors_total;

                return result;
            },
            R"doc(
    Returns a dictionary containing detailed performance and graph statistics.

    Returns:
    --------
    dict:
        A dictionary with the following keys:
        - 'dist_comps': Total number of distance computations.
        - 'search_hops': Total number of nodes visited during searches.
        - 'num_levels': The number of levels in the graph.
        - 'nodes_per_level': A list with the count of nodes on each level.
        - 'links_per_level': A list with the total count of links on each level.
        - 'avg_neighbors_per_level': A list with the average number of neighbors per node for each level.
        - 'avg_neighbors_total': The average number of neighbors per node across the entire graph.
    )doc")
        .def(
            "get_memory_usage",
            [](HnswIndexType& self) -> py::dict {
                py::dict result;
                result["process_rss_bytes"] = getCurrentRSS();
                return result;
            },
            R"doc(
    Returns a dictionary with current memory usage statistics.

    Returns:
    --------
    dict:
        A dictionary with the following keys:
        - 'process_rss_bytes': Total physical memory (RSS) used by the Python process.
    )doc");

    // =========================================================================================
    // DiskANN Bindings Start
    // =========================================================================================

    py::class_<DiskANNBase::BuildParams>(m, "BuildParams")
        .def(py::init<>())
        // R_max_degree is now configured at index creation, so it's removed from BuildParams.
        // It's still a member of the C++ struct, but we don't expose it here to avoid confusion.
        .def_readwrite("L_build", &DiskANNBase::BuildParams::L_build)
        .def_readwrite("alpha", &DiskANNBase::BuildParams::alpha)
        .def_readwrite("num_threads", &DiskANNBase::BuildParams::num_threads);

    py::class_<DiskANNBase::SearchParams>(m, "SearchParams")
        .def(py::init<size_t>(), py::arg("L_search"))
        .def_readwrite("L_search", &DiskANNBase::SearchParams::L_search)
        .def_readwrite("beam_width", &DiskANNBase::SearchParams::beam_width);

    py::class_<DiskANNBase, std::unique_ptr<DiskANNBase>>(m, "DiskANN")
        .def(py::init([](size_t dimensions, uint64_t max_elements, size_t R_max_degree, bool is_dynamic,
                         const std::string& name) {
                 // Ensure system is initialized before creating index
                 initialize_system();
                 
                 uint32_t index_id = 0;
                 
                 // Check if catalog is initialized - if so, create an index entry
                 caliby::IndexCatalog& catalog = caliby::IndexCatalog::instance();
                 if (catalog.is_initialized()) {
                     // Generate a name if not provided
                     std::string idx_name = name.empty() 
                         ? "diskann_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                         : name;
                     
                     // Create catalog entry which sets up per-index file
                     caliby::IndexHandle handle = catalog.create_diskann_index(
                         idx_name, 
                         static_cast<uint32_t>(dimensions),
                         max_elements,
                         static_cast<uint32_t>(R_max_degree),
                         100,  // L_build default
                         1.2f  // alpha default
                     );
                     index_id = handle.index_id();
                 }
                 
                 // Pass the index_id to the factory function.
                 return create_index(dimensions, max_elements, R_max_degree, is_dynamic, index_id);
             }),
             py::arg("dimensions"), py::arg("max_elements"), py::arg("R_max_degree") = 64,
             py::arg("is_dynamic") = false, py::arg("name") = "",
             "Initializes a new, empty Vamana index with a specific dimension, capacity, and max graph degree (R). "
             "If caliby.open() was called, the index will be automatically registered with the catalog and data "
             "will be stored in a per-index file in the catalog directory.")

        .def_property_readonly("dimensions", &DiskANNBase::get_dimensions, "Returns the dimension of the index.")
        // *** ADDED HERE ***
        .def_property_readonly("R", &DiskANNBase::get_R, "Returns the configured max graph degree (R) of the index.")

        .def(
            "build",  // Renamed from "add_items" to "build" for clarity, as it's a bulk-load operation.
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> data,
               const std::vector<std::vector<uint32_t>>& tags, const DiskANNBase::BuildParams& params) {
                if (data.ndim() != 2 || (size_t)data.shape(1) != self.get_dimensions()) {
                    throw std::invalid_argument("Data has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()) + ", but got " +
                                                std::to_string(data.shape(1)));
                }
                size_t num_points = data.shape(0);
                if (tags.size() != num_points) {
                    throw std::invalid_argument("The number of tag lists must match the number of data points.");
                }
                self.build(data.data(), tags, num_points, params);
            },
            py::arg("data"), py::arg("tags"), py::arg("params"),
            "Builds the index from a NumPy array of vectors and a list of tag lists. This is a destructive operation "
            "and can only be called once on a static index.")

        .def(
            "search",  // Renamed for consistency
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, size_t K,
               const DiskANNBase::SearchParams& params) -> py::tuple {
                if (query.ndim() != 1 || (size_t)query.shape(0) != self.get_dimensions()) {
                    throw std::invalid_argument("Query has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()));
                }

                auto result_vec = self.search(query.data(), K, params);

                py::array_t<uint32_t> labels(result_vec.size());
                py::array_t<float> distances(result_vec.size());

                auto labels_ptr = labels.mutable_data();
                auto dists_ptr = distances.mutable_data();

                for (size_t i = 0; i < result_vec.size(); ++i) {
                    labels_ptr[i] = result_vec[i].second;
                    dists_ptr[i] = result_vec[i].first;
                }

                return py::make_tuple(labels, distances);
            },
            py::arg("query"), py::arg("K"), py::arg("params"),
            "Performs a K-Nearest Neighbor search. Returns a tuple of (labels, distances).")

        .def(
            "search_with_filter",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> query,
               uint32_t filter_label, size_t K, const DiskANNBase::SearchParams& params) -> py::tuple {
                if (query.ndim() != 1 || (size_t)query.shape(0) != self.get_dimensions()) {
                    throw std::invalid_argument("Query has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()));
                }

                auto result_vec = self.search_with_filter(query.data(), filter_label, K, params);

                py::array_t<uint32_t> labels(result_vec.size());
                py::array_t<float> distances(result_vec.size());

                auto labels_ptr = labels.mutable_data();
                auto dists_ptr = distances.mutable_data();

                for (size_t i = 0; i < result_vec.size(); ++i) {
                    labels_ptr[i] = result_vec[i].second;
                    dists_ptr[i] = result_vec[i].first;
                }

                return py::make_tuple(labels, distances);
            },
            py::arg("query"), py::arg("filter_label"), py::arg("K"), py::arg("params"),
            "Performs a filtered K-Nearest Neighbor search. Returns a tuple of (labels, distances).")

        .def(
            "search_knn_parallel",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t K,
               const DiskANNBase::SearchParams& params, size_t num_threads) -> py::tuple {
                if (queries.ndim() != 2 || (size_t)queries.shape(1) != self.get_dimensions()) {
                    throw std::invalid_argument("Queries must be a 2D NumPy array with shape (num_queries, dim)");
                }
                size_t num_queries = queries.shape(0);
                if (K == 0) throw std::invalid_argument("K must be > 0");

                auto all_results_vec = self.search_parallel(queries.data(), num_queries, K, params, num_threads);

                // --- Convert result to a pair of NumPy arrays ---
                py::array_t<uint32_t> labels_arr({num_queries, K});
                py::array_t<float> dists_arr({num_queries, K});

                auto labels_ptr = labels_arr.mutable_data();
                auto dists_ptr = dists_arr.mutable_data();

                for (size_t i = 0; i < num_queries; ++i) {
                    for (size_t j = 0; j < K; ++j) {
                        if (j < all_results_vec[i].size()) {
                            labels_ptr[i * K + j] = all_results_vec[i][j].second;
                            dists_ptr[i * K + j] = all_results_vec[i][j].first;
                        } else {
                            labels_ptr[i * K + j] = -1;  // Sentinel for invalid
                            dists_ptr[i * K + j] = std::numeric_limits<float>::infinity();
                        }
                    }
                }
                return py::make_tuple(labels_arr, dists_arr);
            },
            py::arg("queries"), py::arg("K"), py::arg("params"), py::arg("num_threads") = 0,
            "Performs a parallel K-NN search for a batch of queries.")

        .def(
            "search_with_filter_parallel",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries,
               uint32_t filter_label, size_t K, const DiskANNBase::SearchParams& params,
               size_t num_threads) -> py::tuple {
                if (queries.ndim() != 2 || (size_t)queries.shape(1) != self.get_dimensions()) {
                    throw std::invalid_argument("Queries must be a 2D NumPy array with shape (num_queries, dim)");
                }
                size_t num_queries = queries.shape(0);
                if (K == 0) throw std::invalid_argument("K must be > 0");

                auto all_results_vec =
                    self.search_with_filter_parallel(queries.data(), num_queries, filter_label, K, params, num_threads);

                // --- Convert result to a pair of NumPy arrays (same logic as above) ---
                py::array_t<uint32_t> labels_arr({num_queries, K});
                py::array_t<float> dists_arr({num_queries, K});

                auto labels_ptr = labels_arr.mutable_data();
                auto dists_ptr = dists_arr.mutable_data();

                for (size_t i = 0; i < num_queries; ++i) {
                    for (size_t j = 0; j < K; ++j) {
                        if (j < all_results_vec[i].size()) {
                            labels_ptr[i * K + j] = all_results_vec[i][j].second;
                            dists_ptr[i * K + j] = all_results_vec[i][j].first;
                        } else {
                            labels_ptr[i * K + j] = -1;  // Sentinel for invalid
                            dists_ptr[i * K + j] = std::numeric_limits<float>::infinity();
                        }
                    }
                }
                return py::make_tuple(labels_arr, dists_arr);
            },
            py::arg("queries"), py::arg("filter_label"), py::arg("K"), py::arg("params"), py::arg("num_threads") = 0,
            "Performs a parallel filtered K-NN search for a batch of queries.")

        .def(
            "insert_point",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> point,
               const std::vector<uint32_t>& tags, uint32_t external_id) {
                if (point.ndim() != 1 || (size_t)point.shape(0) != self.get_dimensions()) {
                    throw std::invalid_argument("Point has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()));
                }
                self.insert_point(point.data(), tags, external_id);
            },
            py::arg("point"), py::arg("tags"), py::arg("external_id"), "Inserts a single point into a dynamic index.")

        .def("lazy_delete", &DiskANNBase::lazy_delete, py::arg("external_id"),
             "Marks a point for deletion in a dynamic index. The point is removed from search results immediately.")

        .def("consolidate_deletes", &DiskANNBase::consolidate_deletes, py::arg("params"),
             "Repairs the graph structure around deleted nodes. This is a potentially long-running operation.");

    // =========================================================================================
    // IVF+PQ Index Bindings
    // =========================================================================================
    
    using IVFPQIndexType = IVFPQ<L2Distance>;

    py::class_<IVFPQStats>(m, "IVFPQStats")
        .def(py::init<>())
        .def_property_readonly("dist_comps", [](const IVFPQStats& s) { return s.dist_comps.load(); })
        .def_property_readonly("lists_probed", [](const IVFPQStats& s) { return s.lists_probed.load(); })
        .def_property_readonly("vectors_scanned", [](const IVFPQStats& s) { return s.vectors_scanned.load(); })
        .def_readonly("num_clusters", &IVFPQStats::num_clusters)
        .def_readonly("num_subquantizers", &IVFPQStats::num_subquantizers)
        .def_readonly("list_sizes", &IVFPQStats::list_sizes)
        .def_readonly("avg_list_size", &IVFPQStats::avg_list_size)
        .def("__repr__", [](const IVFPQStats& s) {
            return s.toString();
        });

    py::class_<IVFPQIndexType>(m, "IVFPQIndex")
        .def(py::init([](u64 max_elements, size_t dim, u32 num_clusters, u32 num_subquantizers,
                        u32 retrain_interval, bool skip_recovery, uint32_t index_id,
                        const std::string& name) {
            // Ensure system is initialized before creating index
            initialize_system();
            return new IVFPQIndexType(max_elements, dim, num_clusters, num_subquantizers,
                                      retrain_interval, skip_recovery, index_id, name);
        }),
             py::arg("max_elements"),
             py::arg("dim") = 128,
             py::arg("num_clusters") = 256,
             py::arg("num_subquantizers") = 8,
             py::arg("retrain_interval") = 10000,
             py::arg("skip_recovery") = false,
             py::arg("index_id") = 0,
             py::arg("name") = "",
             R"doc(
    Initializes a new IVF+PQ index with runtime parameters.
    
    Parameters:
    -----------
    max_elements : int
        Maximum number of elements the index can hold.
    dim : int
        Dimensionality of vectors (default: 128).
    num_clusters : int
        Number of IVF clusters (K, default: 256).
    num_subquantizers : int
        Number of PQ subquantizers (M, default: 8). dim must be divisible by this.
    retrain_interval : int
        Number of insertions between centroid retraining (default: 10000).
    skip_recovery : bool
        If True, rebuild the index from scratch (default: False).
    index_id : int
        Index ID for multi-index isolation (default: 0).
    name : str
        Optional name identifier for the index (default: "").
)doc")

        .def("flush", [](IVFPQIndexType& self) {
            self.flush();
        }, "Flushes all dirty pages to persistent storage.")
        
        .def("get_name", &IVFPQIndexType::getName,
             "Returns the name of the index.")
        
        .def("get_dim", &IVFPQIndexType::getDim,
             "Returns the dimensionality of vectors managed by the index.")
        
        .def("get_count", [](IVFPQIndexType& self) { return self.size(); },
             "Returns the current number of vectors in the index.")
        
        .def("is_trained", &IVFPQIndexType::isTrained,
             "Returns True if the index has been trained with initial centroids.")
        
        .def("train", [](IVFPQIndexType& self, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> training_data) {
            if (training_data.ndim() != 2) {
                throw std::runtime_error("Training data must be 2-dimensional (n_samples, dim)");
            }
            const size_t dim = self.getDim();
            if (training_data.shape(1) != static_cast<py::ssize_t>(dim)) {
                throw std::runtime_error("Training data has incorrect dimension. Expected " +
                                        std::to_string(dim) + ", but got " + std::to_string(training_data.shape(1)));
            }
            size_t num_samples = training_data.shape(0);
            self.train(training_data.data(), num_samples);
        },
        py::arg("training_data"),
        R"doc(
    Trains the IVF centroids and PQ codebooks using the provided training data.
    Must be called before adding points if the index is not recovered from storage.
    
    Parameters:
    -----------
    training_data : numpy.ndarray
        A 2D NumPy array of shape (n_samples, dim) containing training vectors.
        Recommended to use a representative sample of the dataset (e.g., 10-50k vectors).
)doc")

        .def("add_points", [](IVFPQIndexType& self,
                             py::array_t<float, py::array::c_style | py::array::forcecast> items,
                             size_t num_threads = 0) {
            if (items.ndim() != 2) {
                throw std::runtime_error("Input array must be 2-dimensional (n_samples, n_features)");
            }
            const size_t dim = self.getDim();
            if (items.shape(1) != static_cast<py::ssize_t>(dim)) {
                throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                        std::to_string(dim) + ", but got " + std::to_string(items.shape(1)));
            }
            if (!self.isTrained()) {
                throw std::runtime_error("Index must be trained before adding points. Call train() first.");
            }
            
            size_t n_items = items.shape(0);
            // Generate sequential IDs starting from current size
            std::vector<u32> ids(n_items);
            u64 start_id = self.size();
            for (size_t i = 0; i < n_items; ++i) {
                ids[i] = static_cast<u32>(start_id + i);
            }
            
            self.addPoints(items.data(), ids.data(), n_items, num_threads == 0 ? 1 : num_threads);
        },
        py::arg("items"), py::arg("num_threads") = 0,
        R"doc(
    Adds a batch of points from a NumPy array to the index in parallel.
    
    Parameters:
    -----------
    items : numpy.ndarray
        A 2D NumPy array of shape (num_items, dim) containing the vectors to be added.
    num_threads : int, optional
        The number of threads to use for adding points. If 0, defaults to hardware cores (default: 0).
)doc")

        .def("search_knn", [](IVFPQIndexType& self,
                             py::array_t<float, py::array::c_style | py::array::forcecast> query,
                             size_t k, size_t nprobe, bool stats = false) -> py::object {
            if (query.ndim() != 1) {
                throw std::runtime_error("Query must be a 1-dimensional NumPy array");
            }
            const size_t dim = self.getDim();
            if (query.shape(0) != static_cast<py::ssize_t>(dim)) {
                throw std::runtime_error("Query vector has incorrect dimension. Expected " +
                                        std::to_string(dim) + ", but got " + std::to_string(query.shape(0)));
            }
            
            const float* query_ptr = query.data();
            std::vector<std::pair<float, u32>> result_vec;
            if (stats) {
                result_vec = self.search<true>(query_ptr, k, nprobe);
            } else {
                result_vec = self.search<false>(query_ptr, k, nprobe);
            }
            
            py::list labels;
            py::list distances;
            for (const auto& pair : result_vec) {
                distances.append(pair.first);
                labels.append(pair.second);
            }
            
            return py::make_tuple(py::array(labels), py::array(distances));
        },
        py::arg("query"), py::arg("k"), py::arg("nprobe"), py::arg("stats") = false,
        R"doc(
    Searches for the k-nearest neighbors for a given query vector.
    
    Parameters:
    -----------
    query : numpy.ndarray
        A 1D NumPy array of shape (dim,) containing the query vector.
    k : int
        The number of nearest neighbors to return.
    nprobe : int
        The number of clusters to probe during search (higher = more accurate but slower).
    stats : bool, optional
        If True, update statistics counters (default: False).
    
    Returns:
    --------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple of (labels, distances) arrays.
)doc")

        .def("search_knn_parallel", [](IVFPQIndexType& self,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                       size_t k, size_t nprobe, size_t num_threads = 0,
                                       bool stats = false) -> py::object {
            if (queries.ndim() != 2) {
                throw std::runtime_error("Queries must be a 2-dimensional NumPy array of shape (num_queries, dim)");
            }
            const size_t dim = self.getDim();
            if (queries.shape(1) != static_cast<py::ssize_t>(dim)) {
                throw std::runtime_error("Query vectors have incorrect dimension. Expected " +
                                        std::to_string(dim) + ", but got " + std::to_string(queries.shape(1)));
            }
            if (k == 0) {
                throw std::runtime_error("k must be > 0");
            }
            
            size_t num_queries = queries.shape(0);
            if (num_queries == 0) {
                std::vector<ssize_t> shape = {0, static_cast<ssize_t>(k)};
                py::array_t<int64_t> empty_labels(shape);
                py::array_t<float> empty_distances(shape);
                return py::make_tuple(empty_labels, empty_distances);
            }
            
            std::span<const float> query_span(queries.data(), queries.size());
            std::vector<std::vector<std::pair<float, u32>>> all_results_vec;
            if (stats) {
                all_results_vec = self.searchBatch<true>(query_span, k, nprobe, num_threads);
            } else {
                all_results_vec = self.searchBatch<false>(query_span, k, nprobe, num_threads);
            }
            
            std::vector<ssize_t> result_shape = {static_cast<ssize_t>(num_queries), static_cast<ssize_t>(k)};
            py::array_t<int64_t> labels_arr(result_shape);
            py::array_t<float> distances_arr(result_shape);
            
            auto labels_ptr = labels_arr.mutable_data();
            auto distances_ptr = distances_arr.mutable_data();
            
            for (size_t i = 0; i < num_queries; ++i) {
                const auto& results_for_one_query = all_results_vec[i];
                for (size_t j = 0; j < k; ++j) {
                    if (j < results_for_one_query.size()) {
                        distances_ptr[i * k + j] = results_for_one_query[j].first;
                        labels_ptr[i * k + j] = static_cast<int64_t>(results_for_one_query[j].second);
                    } else {
                        distances_ptr[i * k + j] = std::numeric_limits<float>::infinity();
                        labels_ptr[i * k + j] = -1;
                    }
                }
            }
            
            return py::make_tuple(labels_arr, distances_arr);
        },
        py::arg("queries"), py::arg("k"), py::arg("nprobe"), py::arg("num_threads") = 0,
        py::arg("stats") = false,
        R"doc(
    Searches for the k-nearest neighbors for a batch of query vectors in parallel.
    
    Parameters:
    -----------
    queries : numpy.ndarray
        A 2D NumPy array of shape (num_queries, dim) containing the query vectors.
    k : int
        The number of nearest neighbors to return.
    nprobe : int
        The number of clusters to probe during search.
    num_threads : int, optional
        The number of threads to use for searching. If 0, defaults to hardware cores (default: 0).
    stats : bool, optional
        If True, update statistics counters (default: False).
    
    Returns:
    --------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:
        - labels: Array of shape (num_queries, k) with int64 IDs of nearest neighbors.
        - distances: Array of shape (num_queries, k) with float32 distances.
)doc")

        .def("get_stats", [](IVFPQIndexType& self) -> py::dict {
            IVFPQStats stats = self.getStats();
            py::dict result;
            result["dist_comps"] = stats.dist_comps.load();
            result["lists_probed"] = stats.lists_probed.load();
            result["vectors_scanned"] = stats.vectors_scanned.load();
            result["num_clusters"] = stats.num_clusters;
            result["num_subquantizers"] = stats.num_subquantizers;
            result["list_sizes"] = py::cast(stats.list_sizes);
            result["avg_list_size"] = stats.avg_list_size;
            return result;
        },
        R"doc(
    Returns a dictionary containing performance and index statistics.
    
    Returns:
    --------
    dict:
        A dictionary with the following keys:
        - 'dist_comps': Total number of distance computations.
        - 'lists_probed': Total number of inverted lists probed.
        - 'vectors_scanned': Total number of vectors scanned.
        - 'num_clusters': Number of IVF clusters (K).
        - 'num_subquantizers': Number of PQ subquantizers (M).
        - 'list_sizes': List with the size of each inverted list.
        - 'avg_list_size': Average number of vectors per inverted list.
)doc")

        .def("reset_stats", &IVFPQIndexType::resetStats,
             "Resets live statistics counters.")
        
        .def("get_stats_string", [](IVFPQIndexType& self) {
            return self.getStats().toString();
        }, "Returns a formatted string of current statistics.");

    // =========================================================================================
    // Index Catalog Bindings
    // =========================================================================================

    py::enum_<caliby::IndexType>(m, "IndexType")
        .value("CATALOG", caliby::IndexType::CATALOG)
        .value("HNSW", caliby::IndexType::HNSW)
        .value("DISKANN", caliby::IndexType::DISKANN)
        .value("IVF", caliby::IndexType::IVF)
        .export_values();

    py::enum_<caliby::IndexStatus>(m, "IndexStatus")
        .value("INVALID", caliby::IndexStatus::INVALID)
        .value("CREATING", caliby::IndexStatus::CREATING)
        .value("ACTIVE", caliby::IndexStatus::ACTIVE)
        .value("DELETED", caliby::IndexStatus::DELETED)
        .export_values();

    py::class_<caliby::HNSWConfig>(m, "HNSWConfig")
        .def(py::init<>())
        .def_readwrite("M", &caliby::HNSWConfig::M)
        .def_readwrite("ef_construction", &caliby::HNSWConfig::ef_construction)
        .def_readwrite("max_level", &caliby::HNSWConfig::max_level)
        .def_readwrite("enable_prefetch", &caliby::HNSWConfig::enable_prefetch);

    py::class_<caliby::DiskANNConfig>(m, "DiskANNConfig")
        .def(py::init<>())
        .def_readwrite("R_max_degree", &caliby::DiskANNConfig::R_max_degree)
        .def_readwrite("alpha", &caliby::DiskANNConfig::alpha)
        .def_readwrite("L_build", &caliby::DiskANNConfig::L_build)
        .def_readwrite("is_dynamic", &caliby::DiskANNConfig::is_dynamic);

    py::class_<caliby::IndexConfig>(m, "IndexConfig")
        .def(py::init<>())
        .def_readwrite("dimensions", &caliby::IndexConfig::dimensions)
        .def_readwrite("max_elements", &caliby::IndexConfig::max_elements)
        .def_property("hnsw",
            [](caliby::IndexConfig& self) -> caliby::HNSWConfig& { return self.hnsw; },
            [](caliby::IndexConfig& self, const caliby::HNSWConfig& cfg) { self.hnsw = cfg; })
        .def_property("diskann",
            [](caliby::IndexConfig& self) -> caliby::DiskANNConfig& { return self.diskann; },
            [](caliby::IndexConfig& self, const caliby::DiskANNConfig& cfg) { self.diskann = cfg; });

    py::class_<caliby::IndexInfo>(m, "IndexInfo")
        .def_readonly("index_id", &caliby::IndexInfo::index_id)
        .def_readonly("name", &caliby::IndexInfo::name)
        .def_readonly("type", &caliby::IndexInfo::type)
        .def_readonly("status", &caliby::IndexInfo::status)
        .def_readonly("dimensions", &caliby::IndexInfo::dimensions)
        .def_readonly("max_elements", &caliby::IndexInfo::max_elements)
        .def_readonly("num_elements", &caliby::IndexInfo::num_elements)
        .def_readonly("create_time", &caliby::IndexInfo::create_time)
        .def_readonly("modify_time", &caliby::IndexInfo::modify_time)
        .def_readonly("file_path", &caliby::IndexInfo::file_path)
        .def("__repr__", [](const caliby::IndexInfo& info) {
            return "<IndexInfo name='" + info.name + "' type=" + 
                   std::to_string(static_cast<int>(info.type)) +
                   " elements=" + std::to_string(info.num_elements) + ">";
        });

    py::class_<caliby::IndexHandle>(m, "IndexHandle")
        .def("is_valid", &caliby::IndexHandle::is_valid)
        .def("index_id", &caliby::IndexHandle::index_id)
        .def("name", &caliby::IndexHandle::name)
        .def("type", &caliby::IndexHandle::type)
        .def("dimensions", &caliby::IndexHandle::dimensions)
        .def("max_elements", &caliby::IndexHandle::max_elements)
        .def("flush", &caliby::IndexHandle::flush)
        .def("global_page_id", &caliby::IndexHandle::global_page_id)
        .def("__repr__", [](const caliby::IndexHandle& h) {
            return "<IndexHandle name='" + h.name() + "' id=" + 
                   std::to_string(h.index_id()) + ">";
        });

    py::class_<caliby::IndexCatalog, std::unique_ptr<caliby::IndexCatalog, py::nodelete>>(m, "IndexCatalog")
        .def_static("instance", &caliby::IndexCatalog::instance, py::return_value_policy::reference,
             "Get the singleton IndexCatalog instance.")
        .def("initialize", &caliby::IndexCatalog::initialize, py::arg("data_dir"), py::arg("cleanup_if_exist") = false,
             "Initialize the catalog in the specified directory.")
        .def("is_initialized", &caliby::IndexCatalog::is_initialized,
             "Check if the catalog is initialized.")
        .def("shutdown", &caliby::IndexCatalog::shutdown,
             "Shutdown the catalog and close all files.")
        .def("create_index",
             py::overload_cast<const std::string&, caliby::IndexType, const caliby::IndexConfig&>(&caliby::IndexCatalog::create_index),
             py::arg("name"), py::arg("type"), py::arg("config"),
             "Create a new index with the given name, type, and configuration.")
        .def("create_hnsw_index", &caliby::IndexCatalog::create_hnsw_index,
             py::arg("name"), py::arg("dimensions"), py::arg("max_elements"),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             "Create a new HNSW index with simplified parameters.")
        .def("create_diskann_index", &caliby::IndexCatalog::create_diskann_index,
             py::arg("name"), py::arg("dimensions"), py::arg("max_elements"),
             py::arg("R_max_degree") = 64, py::arg("L_build") = 100, py::arg("alpha") = 1.2f,
             "Create a new DiskANN index with simplified parameters.")
        .def("open_index", &caliby::IndexCatalog::open_index, py::arg("name"),
             "Open an existing index by name.")
        .def("drop_index", &caliby::IndexCatalog::drop_index, py::arg("name"),
             "Drop (delete) an index by name.")
        .def("index_exists", &caliby::IndexCatalog::index_exists, py::arg("name"),
             "Check if an index with the given name exists.")
        .def("list_indexes", &caliby::IndexCatalog::list_indexes,
             "List all active indexes.")
        .def("get_index_info", &caliby::IndexCatalog::get_index_info, py::arg("name"),
             "Get detailed information about an index.")
        .def("data_dir", &caliby::IndexCatalog::data_dir,
             "Get the data directory path.");

    // =========================================================================================
    // Collection Bindings
    // =========================================================================================

    py::enum_<caliby::FieldType>(m, "FieldType")
        .value("STRING", caliby::FieldType::STRING)
        .value("INT", caliby::FieldType::INT)
        .value("FLOAT", caliby::FieldType::FLOAT)
        .value("BOOL", caliby::FieldType::BOOL)
        .value("STRING_ARRAY", caliby::FieldType::STRING_ARRAY)
        .value("INT_ARRAY", caliby::FieldType::INT_ARRAY)
        .export_values();

    py::class_<caliby::FieldDef>(m, "FieldDef")
        .def(py::init<>())
        .def(py::init<const std::string&, caliby::FieldType, bool>(),
             py::arg("name"), py::arg("type"), py::arg("nullable") = true)
        .def_readwrite("name", &caliby::FieldDef::name)
        .def_readwrite("type", &caliby::FieldDef::type)
        .def_readwrite("nullable", &caliby::FieldDef::nullable);

    py::class_<caliby::Schema>(m, "Schema")
        .def(py::init<>())
        .def("add_field", &caliby::Schema::add_field,
             py::arg("name"), py::arg("type"), py::arg("nullable") = true,
             "Add a field to the schema.")
        .def("has_field", &caliby::Schema::has_field, py::arg("name"),
             "Check if a field exists in the schema.")
        .def("get_field", &caliby::Schema::get_field, py::arg("name"),
             py::return_value_policy::reference,
             "Get a field definition by name.")
        .def("fields", &caliby::Schema::fields,
             "Get all field definitions.");

    py::class_<caliby::Document>(m, "Document")
        .def(py::init<>())
        .def(py::init([](uint64_t id, const std::string& content, const py::object& meta) {
            caliby::Document doc;
            doc.id = id;
            doc.content = content;
            if (!meta.is_none()) {
                py::module_ json_module = py::module_::import("json");
                std::string json_str = py::cast<std::string>(json_module.attr("dumps")(meta));
                doc.metadata = nlohmann::json::parse(json_str);
            } else {
                doc.metadata = nlohmann::json::object();
            }
            return doc;
        }), py::arg("id"), py::arg("content") = "", py::arg("metadata") = py::none())
        .def_readwrite("id", &caliby::Document::id)
        .def_readwrite("content", &caliby::Document::content)
        .def_property("metadata",
            [](const caliby::Document& d) {
                // Convert nlohmann::json to Python dict
                py::module_ json_module = py::module_::import("json");
                std::string json_str = d.metadata.dump();
                return json_module.attr("loads")(json_str);
            },
            [](caliby::Document& d, const py::object& meta) {
                // Convert Python dict to nlohmann::json via string serialization
                py::module_ json_module = py::module_::import("json");
                std::string json_str = py::cast<std::string>(json_module.attr("dumps")(meta));
                d.metadata = nlohmann::json::parse(json_str);
            },
            "Document metadata as a dictionary.")
        .def("__repr__", [](const caliby::Document& d) {
            return "<Document id=" + std::to_string(d.id) + ">";
        });

    py::class_<caliby::SearchResult>(m, "SearchResult")
        .def_readonly("doc_id", &caliby::SearchResult::doc_id)
        .def_readonly("score", &caliby::SearchResult::score)
        .def_readonly("vector_score", &caliby::SearchResult::vector_score)
        .def_readonly("text_score", &caliby::SearchResult::text_score)
        .def_property_readonly("id", [](const caliby::SearchResult& r) {
            return r.doc_id;  // Alias for doc_id
        })
        .def_property_readonly("document", [](const caliby::SearchResult& r) -> py::object {
            if (!r.document.has_value()) {
                return py::none();
            }
            // Create a Python dict representing the document
            py::module_ json_module = py::module_::import("json");
            py::dict d;
            d["id"] = r.document->id;
            d["content"] = r.document->content;
            std::string meta_str = r.document->metadata.dump();
            d["metadata"] = json_module.attr("loads")(meta_str);
            return d;
        })
        .def("__repr__", [](const caliby::SearchResult& r) {
            return "<SearchResult doc_id=" + std::to_string(r.doc_id) +
                   " score=" + std::to_string(r.score) + ">";
        });

    py::enum_<caliby::FusionMethod>(m, "FusionMethod")
        .value("RRF", caliby::FusionMethod::RRF)
        .value("WEIGHTED", caliby::FusionMethod::WEIGHTED)
        .export_values();

    py::enum_<caliby::DistanceMetric>(m, "DistanceMetric")
        .value("L2", caliby::DistanceMetric::L2)
        .value("COSINE", caliby::DistanceMetric::COSINE)
        .value("IP", caliby::DistanceMetric::IP)
        .export_values();

    py::class_<caliby::FusionParams>(m, "FusionParams")
        .def(py::init<>())
        .def_readwrite("method", &caliby::FusionParams::method)
        .def_readwrite("rrf_k", &caliby::FusionParams::rrf_k)
        .def_readwrite("vector_weight", &caliby::FusionParams::vector_weight)
        .def_readwrite("text_weight", &caliby::FusionParams::text_weight);

    py::class_<caliby::CollectionIndexInfo>(m, "CollectionIndexInfo")
        .def_readonly("index_id", &caliby::CollectionIndexInfo::index_id)
        .def_readonly("name", &caliby::CollectionIndexInfo::name)
        .def_readonly("type", &caliby::CollectionIndexInfo::type)
        .def_readonly("status", &caliby::CollectionIndexInfo::status)
        .def_property_readonly("config", [](const caliby::CollectionIndexInfo& info) {
            // Convert nlohmann::json to Python dict
            py::module_ json_module = py::module_::import("json");
            std::string json_str = info.config.dump();
            return json_module.attr("loads")(json_str);
        })
        .def("__repr__", [](const caliby::CollectionIndexInfo& i) {
            return "<CollectionIndexInfo name='" + i.name + "' type='" + i.type + "'>";
        });

    py::class_<caliby::Collection>(m, "Collection")
        .def(py::init([](const std::string& name, const caliby::Schema& schema, 
                         uint32_t vector_dim, caliby::DistanceMetric distance_metric) {
            // Ensure system is initialized
            initialize_system();
            return new caliby::Collection(name, schema, vector_dim, distance_metric);
        }), py::arg("name"), py::arg("schema"), py::arg("vector_dim") = 0,
           py::arg("distance_metric") = caliby::DistanceMetric::COSINE,
        "Create a new collection with the given name and schema.")
        
        .def_static("open", &caliby::Collection::open, py::arg("name"),
             "Open an existing collection by name.")
        
        .def("name", &caliby::Collection::name, "Get the collection name.")
        .def("schema", &caliby::Collection::schema, py::return_value_policy::reference,
             "Get the collection schema.")
        .def("doc_count", &caliby::Collection::doc_count, "Get the number of documents.")
        .def("vector_dim", &caliby::Collection::vector_dim, "Get vector dimensions.")
        .def("has_vectors", &caliby::Collection::has_vectors, "Check if collection supports vectors.")
        
        // Document operations (batch-oriented API)
        .def("add", [](caliby::Collection& self, 
                       const std::vector<std::string>& contents,
                       const py::list& metadatas,
                       const std::vector<std::vector<float>>& vectors) {
            py::module_ json_module = py::module_::import("json");
            std::vector<nlohmann::json> metas;
            for (const auto& m : metadatas) {
                std::string json_str = py::cast<std::string>(json_module.attr("dumps")(m));
                metas.push_back(nlohmann::json::parse(json_str));
            }
            return self.add(contents, metas, vectors);
        }, py::arg("contents"), py::arg("metadatas"),
           py::arg("vectors") = std::vector<std::vector<float>>{},
        "Add documents to the collection. Returns assigned document IDs.")
        
        .def("get", py::overload_cast<const std::vector<uint64_t>&>(&caliby::Collection::get),
             py::arg("ids"), "Get documents by IDs.")
        
        .def("update", [](caliby::Collection& self,
                          const std::vector<uint64_t>& ids,
                          const py::list& metadatas) {
            py::module_ json_module = py::module_::import("json");
            std::vector<nlohmann::json> metas;
            for (const auto& m : metadatas) {
                std::string json_str = py::cast<std::string>(json_module.attr("dumps")(m));
                metas.push_back(nlohmann::json::parse(json_str));
            }
            self.update(ids, metas);
        }, py::arg("ids"), py::arg("metadatas"),
        "Update document metadata.")
        
        .def("delete", py::overload_cast<const std::vector<uint64_t>&>(&caliby::Collection::delete_docs),
             py::arg("ids"), "Delete documents by IDs.")
        
        // Index creation
        .def("create_hnsw_index", &caliby::Collection::create_hnsw_index,
             py::arg("name"), py::arg("M") = 16, py::arg("ef_construction") = 200,
             "Create an HNSW index for vector search.")
        
        .def("create_diskann_index", &caliby::Collection::create_diskann_index,
             py::arg("name"), py::arg("R") = 64, py::arg("L") = 100, py::arg("alpha") = 1.2f,
             "Create a DiskANN index for vector search.")
        
        .def("create_text_index", [](caliby::Collection& self, const std::string& name) {
            self.create_text_index(name, caliby::TextIndexConfig{});
        }, py::arg("name"),
        "Create a text index with BM25 scoring.")
        
        // New API: create_metadata_index with support for composite indices
        .def("create_metadata_index", [](caliby::Collection& self, const std::string& name,
                                          const std::vector<std::string>& fields, bool unique) {
            caliby::MetadataIndexConfig config(fields, unique);
            self.create_metadata_index(name, config);
        }, py::arg("name"), py::arg("fields"), py::arg("unique") = false,
        R"doc(Create a metadata index on one or more fields.

Supports composite indices with leftmost prefix rule (like MySQL secondary indices).

Args:
    name: Index name
    fields: List of field names to index. For composite indices, order matters.
    unique: Whether the full composite key must be unique (default: False)

Examples:
    # Single-field index
    collection.create_metadata_index("year_idx", ["year"])
    
    # Composite index - can efficiently query:
    #   - category = 'tech'
    #   - category = 'tech' AND year = 2024
    # Cannot efficiently query:
    #   - year = 2024 (leftmost field missing)
    collection.create_metadata_index("category_year_idx", ["category", "year"])
)doc")
        
        // Legacy API: create_btree_index (single field only, for backward compatibility)
        .def("create_btree_index", [](caliby::Collection& self, const std::string& name,
                                       const std::string& field, bool unique) {
            caliby::MetadataIndexConfig config({field}, unique);
            self.create_metadata_index(name, config);
        }, py::arg("name"), py::arg("field"), py::arg("unique") = false,
        "Create a B-tree index on a metadata field. (Legacy API - use create_metadata_index instead)")
        
        .def("list_indices", [](caliby::Collection& self) {
            py::list result;
            py::module_ json_module = py::module_::import("json");
            for (const auto& info : self.list_indices()) {
                py::dict d;
                d["index_id"] = info.index_id;
                d["name"] = info.name;
                d["type"] = info.type;
                d["status"] = info.status;
                std::string config_str = info.config.dump();
                d["config"] = json_module.attr("loads")(config_str);
                result.append(d);
            }
            return result;
        }, "List all indices in the collection.")
        
        .def("drop_index", &caliby::Collection::drop_index, py::arg("name"),
             "Drop an index by name.")
        
        // Search operations
        .def("search_vector", [](caliby::Collection& self,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> query,
                                  const std::string& index_name,
                                  size_t k,
                                  const std::string& filter_json) {
            std::vector<float> q(query.data(), query.data() + query.size());
            std::optional<caliby::FilterCondition> filter;
            if (!filter_json.empty()) {
                filter = caliby::FilterCondition::from_json(nlohmann::json::parse(filter_json));
            }
            return self.search_vector(q, index_name, k, filter);
        }, py::arg("query"), py::arg("index_name"), py::arg("k"), py::arg("filter") = "",
        "Search for similar vectors. Optional filter as JSON string.")
        
        .def("search_text", [](caliby::Collection& self,
                               const std::string& query,
                               const std::string& index_name,
                               size_t k,
                               const std::string& filter_json) {
            std::optional<caliby::FilterCondition> filter;
            if (!filter_json.empty()) {
                filter = caliby::FilterCondition::from_json(nlohmann::json::parse(filter_json));
            }
            return self.search_text(query, index_name, k, filter);
        }, py::arg("query"), py::arg("index_name"), py::arg("k"), py::arg("filter") = "",
        "Search text using BM25 scoring. Optional filter as JSON string.")
        
        .def("search_hybrid", [](caliby::Collection& self,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> query_vec,
                                  const std::string& vector_index,
                                  const std::string& query_text,
                                  const std::string& text_index,
                                  size_t k,
                                  const caliby::FusionParams& fusion,
                                  const std::string& filter_json) {
            std::vector<float> q(query_vec.data(), query_vec.data() + query_vec.size());
            std::optional<caliby::FilterCondition> filter;
            if (!filter_json.empty()) {
                filter = caliby::FilterCondition::from_json(nlohmann::json::parse(filter_json));
            }
            return self.search_hybrid(q, vector_index, query_text, text_index, k, fusion, filter);
        }, py::arg("query_vec"), py::arg("vector_index"),
           py::arg("query_text"), py::arg("text_index"),
           py::arg("k"), py::arg("fusion") = caliby::FusionParams{},
           py::arg("filter") = "",
        "Perform hybrid vector + text search with score fusion.")
        
        .def("flush", &caliby::Collection::flush, "Flush all changes to storage.");

    // FilterCondition helper for building filters in Python
    m.def("make_filter", [](const std::string& json) {
        return caliby::FilterCondition::from_json(nlohmann::json::parse(json));
    }, py::arg("json"), 
    R"doc(
Create a filter condition from a JSON string.

Filter DSL examples:
  - {"field": "age", "op": "gt", "value": 18}
  - {"field": "status", "op": "eq", "value": "active"}
  - {"and": [{"field": "age", "op": "gte", "value": 21}, {"field": "country", "op": "eq", "value": "US"}]}
  - {"or": [{"field": "category", "op": "eq", "value": "A"}, {"field": "category", "op": "eq", "value": "B"}]}
  - {"not": {"field": "deleted", "op": "eq", "value": true}}

Supported operators: eq, ne, gt, gte, lt, lte, in, contains
)doc");
}