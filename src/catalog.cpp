/**
 * @file catalog.cpp
 * @brief Implementation of Caliby Index Catalog System
 */

#include "catalog.hpp"
#include "calico.hpp"
#include "logging.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace caliby {

namespace fs = std::filesystem;

//=============================================================================
// MultiFileStorage Implementation
//=============================================================================

MultiFileStorage::~MultiFileStorage() {
    std::unique_lock lock(fd_mutex_);
    for (auto& [index_id, fd] : index_to_fd_) {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    index_to_fd_.clear();
}

void MultiFileStorage::initialize(const std::string& data_dir) {
    // Close any open files from previous session
    {
        std::unique_lock lock(fd_mutex_);
        for (auto& [index_id, fd] : index_to_fd_) {
            if (fd >= 0) {
                ::close(fd);
            }
        }
        index_to_fd_.clear();
    }
    
    data_dir_ = data_dir;
    
    // Create directory if it doesn't exist
    if (!fs::exists(data_dir_)) {
        fs::create_directories(data_dir_);
    }
}

int MultiFileStorage::open_file(uint32_t index_id, const std::string& filename, bool create) {
    std::unique_lock lock(fd_mutex_);
    
    // Check if already open
    auto it = index_to_fd_.find(index_id);
    if (it != index_to_fd_.end() && it->second >= 0) {
        return it->second;
    }
    
    std::string full_path = data_dir_ + "/" + filename;
    
    // Catalog file should NOT use O_DIRECT due to small, potentially unaligned I/O
    bool is_catalog = (index_id == CATALOG_INDEX_ID);
    
    int flags = O_RDWR;
    if (create) {
        flags |= O_CREAT;
    }
    
    // Use O_DIRECT for index files only (not for catalog)
    #ifdef __linux__
    if (!is_catalog) {
        flags |= O_DIRECT;
    }
    #endif
    
    int fd = ::open(full_path.c_str(), flags, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        if (!create) {
            throw std::runtime_error("Failed to open index file: " + full_path + 
                                    " (errno: " + std::to_string(errno) + ")");
        }
        // Try without O_DIRECT for creation
        flags &= ~O_DIRECT;
        fd = ::open(full_path.c_str(), flags, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            throw std::runtime_error("Failed to create index file: " + full_path +
                                    " (errno: " + std::to_string(errno) + ")");
        }
        // Reopen with O_DIRECT for index files only
        ::close(fd);
        #ifdef __linux__
        if (!is_catalog) {
            flags |= O_DIRECT;
        }
        #endif
        fd = ::open(full_path.c_str(), flags, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            // Fall back to non-direct I/O
            flags &= ~O_DIRECT;
            fd = ::open(full_path.c_str(), flags, S_IRUSR | S_IWUSR);
        }
    }
    
    index_to_fd_[index_id] = fd;
    return fd;
}

void MultiFileStorage::close_file(uint32_t index_id) {
    std::unique_lock lock(fd_mutex_);
    
    auto it = index_to_fd_.find(index_id);
    if (it != index_to_fd_.end()) {
        if (it->second >= 0) {
            ::close(it->second);
        }
        index_to_fd_.erase(it);
    }
}

int MultiFileStorage::get_fd(uint32_t index_id) const {
    std::shared_lock lock(fd_mutex_);
    
    auto it = index_to_fd_.find(index_id);
    if (it != index_to_fd_.end()) {
        return it->second;
    }
    return -1;
}

void MultiFileStorage::read_page(uint32_t index_id, uint64_t local_page_id, void* buffer) {
    int fd = get_fd(index_id);
    if (fd < 0) {
        throw std::runtime_error("Index file not open: " + std::to_string(index_id));
    }
    
    off_t offset = static_cast<off_t>(local_page_id) * PAGE_SIZE;
    ssize_t bytes_read = ::pread(fd, buffer, PAGE_SIZE, offset);
    
    if (bytes_read < 0) {
        throw std::runtime_error("Failed to read page: " + std::to_string(local_page_id) +
                                " (errno: " + std::to_string(errno) + ")");
    }
    
    // Zero-fill if file is shorter
    if (static_cast<size_t>(bytes_read) < PAGE_SIZE) {
        std::memset(static_cast<char*>(buffer) + bytes_read, 0, PAGE_SIZE - bytes_read);
    }
}

void MultiFileStorage::write_page(uint32_t index_id, uint64_t local_page_id, const void* buffer) {
    int fd = get_fd(index_id);
    if (fd < 0) {
        throw std::runtime_error("Index file not open: " + std::to_string(index_id));
    }
    
    off_t offset = static_cast<off_t>(local_page_id) * PAGE_SIZE;
    ssize_t bytes_written = ::pwrite(fd, buffer, PAGE_SIZE, offset);
    
    if (bytes_written != static_cast<ssize_t>(PAGE_SIZE)) {
        throw std::runtime_error("Failed to write page: " + std::to_string(local_page_id) +
                                " (errno: " + std::to_string(errno) + ")");
    }
}

std::string MultiFileStorage::make_index_filename(IndexType type, uint32_t index_id, const std::string& index_name) const {
    std::string type_str;
    switch (type) {
        case IndexType::HNSW:       type_str = "hnsw"; break;
        case IndexType::DISKANN:    type_str = "diskann"; break;
        case IndexType::IVF:        type_str = "ivf"; break;
        case IndexType::COLLECTION: type_str = "collection"; break;
        case IndexType::TEXT:       type_str = "text"; break;
        case IndexType::BTREE:      type_str = "btree"; break;
        default:                    type_str = "unknown"; break;
    }
    return "caliby_" + type_str + "_" + std::to_string(index_id) + "_" + index_name + ".dat";
}

bool MultiFileStorage::delete_file(const std::string& filename) {
    std::string full_path = data_dir_ + "/" + filename;
    return fs::remove(full_path);
}

//=============================================================================
// IndexHandle Implementation
//=============================================================================

IndexHandle::IndexHandle(IndexCatalog* catalog, uint32_t index_id, int file_fd,
                        const std::string& name, IndexType type, uint32_t dimensions,
                        uint64_t max_elements)
    : catalog_(catalog)
    , index_id_(index_id)
    , file_fd_(file_fd)
    , name_(name)
    , type_(type)
    , dimensions_(dimensions)
    , max_elements_(max_elements)
{}

IndexHandle::IndexHandle(IndexHandle&& other) noexcept
    : catalog_(other.catalog_)
    , index_id_(other.index_id_)
    , file_fd_(other.file_fd_)
    , name_(std::move(other.name_))
    , type_(other.type_)
    , dimensions_(other.dimensions_)
    , max_elements_(other.max_elements_)
{
    other.catalog_ = nullptr;
    other.index_id_ = 0;
    other.file_fd_ = -1;
}

IndexHandle& IndexHandle::operator=(IndexHandle&& other) noexcept {
    if (this != &other) {
        catalog_ = other.catalog_;
        index_id_ = other.index_id_;
        file_fd_ = other.file_fd_;
        name_ = std::move(other.name_);
        type_ = other.type_;
        dimensions_ = other.dimensions_;
        max_elements_ = other.max_elements_;
        
        other.catalog_ = nullptr;
        other.index_id_ = 0;
        other.file_fd_ = -1;
    }
    return *this;
}

IndexHandle::~IndexHandle() {
    // Note: We don't close the file here because the catalog manages file lifecycles
}

BufferManager* IndexHandle::buffer_manager() const {
    return catalog_ ? catalog_->buffer_manager() : nullptr;
}

uint64_t IndexHandle::allocate_page() {
    // This would be implemented when integrating with the buffer manager
    // For now, return 0 as placeholder
    return 0;
}

void IndexHandle::update_element_count(uint64_t count) {
    if (catalog_) {
        catalog_->update_index_element_count(index_id_, count);
    }
}

void IndexHandle::flush() {
    if (file_fd_ >= 0) {
        ::fsync(file_fd_);
    }
}

//=============================================================================
// IndexCatalog Implementation
//=============================================================================

IndexCatalog& IndexCatalog::instance() {
    static IndexCatalog instance;
    return instance;
}

IndexCatalog::~IndexCatalog() {
    shutdown();
}

void IndexCatalog::initialize(const std::string& data_dir, bool cleanup_if_exist) {
    std::unique_lock lock(catalog_mutex_);
    
    if (initialized_.load()) {
        return;  // Already initialized
    }
    
    // Handle cleanup if requested
    if (cleanup_if_exist && fs::exists(data_dir)) {
        CALIBY_LOG_INFO("IndexCatalog", "Cleaning up existing directory: ", data_dir);
        // Remove all contents but keep the directory
        for (const auto& entry : fs::directory_iterator(data_dir)) {
            fs::remove_all(entry.path());
        }
        CALIBY_LOG_INFO("IndexCatalog", "Cleanup complete");
    }
    
    // Create data directory if it doesn't exist
    if (!fs::exists(data_dir)) {
        fs::create_directories(data_dir);
    }
    
    // Acquire exclusive lock on the data directory
    std::string lock_path = data_dir + "/.caliby.lock";
    lock_fd_ = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (lock_fd_ < 0) {
        throw std::runtime_error("Failed to open lock file: " + std::string(strerror(errno)));
    }
    
    // Try to acquire exclusive lock (non-blocking)
    struct flock fl;
    fl.l_type = F_WRLCK;    // Exclusive write lock
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;           // Lock entire file
    
    if (fcntl(lock_fd_, F_SETLK, &fl) == -1) {
        close(lock_fd_);
        lock_fd_ = -1;
        if (errno == EACCES || errno == EAGAIN) {
            throw std::runtime_error("Data directory is already locked by another process: " + data_dir);
        } else {
            throw std::runtime_error("Failed to lock data directory: " + std::string(strerror(errno)));
        }
    }
    
    CALIBY_LOG_DEBUG("IndexCatalog", "Acquired exclusive lock on: ", data_dir);
    
    // Initialize storage (this will close any files from previous session)
    storage_.initialize(data_dir);
    
    // Check if catalog file exists
    std::string catalog_path = data_dir + "/caliby_catalog";
    bool catalog_exists = fs::exists(catalog_path);
    
    if (catalog_exists) {
        CALIBY_LOG_INFO("IndexCatalog", "Found existing catalog, performing recovery...");
        
        // Load existing catalog
        load_catalog();
        
        CALIBY_LOG_INFO("IndexCatalog", "Loaded ", entries_.size(), " index entries from catalog");
        
        // Re-register all loaded indexes with BufferManager
        if (buffer_manager_) {
            size_t recovered = 0;
            for (const auto& entry : entries_) {
                if (entry.is_active()) {
                    try {
                        // Open the file to get fd
                        int fd = storage_.open_file(entry.index_id, entry.file_path, false);
                        
                        // Use stored alloc_pages if available, otherwise estimate
                        uint64_t initial_pages;
                        uint64_t initial_alloc_count = 0;
                        if (entry.alloc_pages > 0) {
                            // Use the actual allocated pages from last run
                            initial_pages = entry.alloc_pages;
                            initial_alloc_count = entry.alloc_pages;  // Also restore allocCount
                        } else if (entry.max_elements == 0) {
                            initial_pages = 1024;  // Unbounded - start small, grows automatically
                        } else {
                            initial_pages = (entry.max_elements / 2) + 1024;
                        }
                        buffer_manager_->registerIndex(entry.index_id, initial_pages, initial_alloc_count, fd);
                        recovered++;
                        
                        CALIBY_LOG_DEBUG("IndexCatalog", "Recovered index: ", entry.name, 
                                        " (id=", entry.index_id, ")");
                    } catch (const std::exception& e) {
                        CALIBY_LOG_WARN("IndexCatalog", "Failed to recover index ", entry.name, 
                                       ": ", e.what());
                    }
                }
            }
            CALIBY_LOG_INFO("IndexCatalog", "Recovery complete: ", recovered, " indexes recovered");
        }
    } else {
        CALIBY_LOG_INFO("IndexCatalog", "Creating new catalog in: ", data_dir);
        
        // Create new catalog
        header_.initialize();
        entries_.clear();
        name_to_index_id_.clear();
        
        // Create catalog file and write header
        storage_.open_file(CATALOG_INDEX_ID, "caliby_catalog", true);
        save_catalog_header();
    }
    
    // Open catalog file if not already open
    if (storage_.get_fd(CATALOG_INDEX_ID) < 0) {
        storage_.open_file(CATALOG_INDEX_ID, "caliby_catalog", false);
    }
    
    initialized_.store(true);
    
    CALIBY_LOG_INFO("IndexCatalog", "Initialized in: ", data_dir);
    CALIBY_LOG_INFO("IndexCatalog", "Found ", header_.num_indexes, " existing indexes");
}

void IndexCatalog::shutdown() {
    std::unique_lock lock(catalog_mutex_);
    
    if (!initialized_.load()) {
        return;
    }
    
    // Save catalog state before clearing
    save_catalog_header();
    
    // Close all files (handled by MultiFileStorage destructor when storage_ goes out of scope)
    // But we need to explicitly clear our state
    entries_.clear();
    name_to_index_id_.clear();
    
    // Release the directory lock
    if (lock_fd_ >= 0) {
        struct flock fl;
        fl.l_type = F_UNLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0;
        fcntl(lock_fd_, F_SETLK, &fl);
        close(lock_fd_);
        lock_fd_ = -1;
        CALIBY_LOG_DEBUG("IndexCatalog", "Released directory lock");
    }
    
    // Do NOT reset header_ here - it should persist for the saved catalog file
    // header_ will be reloaded from disk on next initialize()
    
    initialized_.store(false);
    
    CALIBY_LOG_INFO("IndexCatalog", "Shutdown complete");
}

void IndexCatalog::load_catalog() {
    // Allocate aligned buffer for reading
    alignas(4096) char buffer[4096];
    
    // Open and read catalog file
    int fd = storage_.open_file(CATALOG_INDEX_ID, "caliby_catalog", false);
    if (fd < 0) {
        throw std::runtime_error("Failed to open catalog file");
    }
    
    // Read header (page 0)
    storage_.read_page(CATALOG_INDEX_ID, 0, buffer);
    std::memcpy(&header_, buffer, sizeof(CatalogHeader));
    
    if (!header_.is_valid()) {
        throw std::runtime_error("Invalid catalog file: bad magic or version");
    }
    
    // Read index entries
    entries_.clear();
    name_to_index_id_.clear();
    
    uint32_t entries_read = 0;
    uint64_t page_id = 1;  // Entries start at page 1
    
    while (entries_read < header_.num_indexes) {
        storage_.read_page(CATALOG_INDEX_ID, page_id, buffer);
        
        auto* entry_page = reinterpret_cast<IndexEntryPage*>(buffer);
        
        for (size_t i = 0; i < ENTRIES_PER_PAGE && entries_read < header_.num_indexes; ++i) {
            const auto& entry = entry_page->entries[i];
            if (entry.is_valid() && entry.is_active()) {
                entries_.push_back(entry);
                name_to_index_id_[entry.name] = entry.index_id;
                entries_read++;
            }
        }
        
        page_id++;
    }
    
    CALIBY_LOG_INFO("IndexCatalog", "Loaded ", entries_.size(), " indexes from catalog");
}

void IndexCatalog::save_catalog_header() {
    alignas(4096) char buffer[4096];
    std::memset(buffer, 0, sizeof(buffer));
    std::memcpy(buffer, &header_, sizeof(CatalogHeader));
    
    storage_.write_page(CATALOG_INDEX_ID, 0, buffer);
    
    // Also sync to disk
    int fd = storage_.get_fd(CATALOG_INDEX_ID);
    if (fd >= 0) {
        ::fsync(fd);
    }
}

void IndexCatalog::save_index_entry(const IndexEntry& entry) {
    // Find the page and slot for this entry
    // For simplicity, we write entries sequentially
    
    alignas(4096) char buffer[4096];
    
    // Find which page this entry belongs to
    size_t entry_idx = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].index_id == entry.index_id) {
            entry_idx = i;
            break;
        }
    }
    
    uint64_t page_id = 1 + (entry_idx / ENTRIES_PER_PAGE);
    size_t slot = entry_idx % ENTRIES_PER_PAGE;
    
    // Read current page
    storage_.read_page(CATALOG_INDEX_ID, page_id, buffer);
    
    // Update entry
    auto* entry_page = reinterpret_cast<IndexEntryPage*>(buffer);
    entry_page->entries[slot] = entry;
    
    // Write back
    storage_.write_page(CATALOG_INDEX_ID, page_id, buffer);
}

IndexEntry* IndexCatalog::find_entry_by_id(uint32_t index_id) {
    for (auto& entry : entries_) {
        if (entry.index_id == index_id) {
            return &entry;
        }
    }
    return nullptr;
}

IndexEntry* IndexCatalog::find_entry_by_name(const std::string& name) {
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        return nullptr;
    }
    return find_entry_by_id(it->second);
}

uint32_t IndexCatalog::allocate_index_id() {
    return header_.next_index_id++;
}

IndexHandle IndexCatalog::create_index(const std::string& name, IndexType type,
                                       const IndexConfig& config) {
    std::unique_lock lock(catalog_mutex_);
    
    if (!initialized_.load()) {
        throw std::runtime_error("Catalog not initialized");
    }
    
    // Check name doesn't exist
    if (name_to_index_id_.find(name) != name_to_index_id_.end()) {
        throw std::runtime_error("Index already exists: " + name);
    }
    
    // Validate name
    if (name.empty() || name.length() >= MAX_INDEX_NAME_LEN) {
        throw std::runtime_error("Invalid index name length");
    }
    
    // Allocate index ID
    uint32_t index_id = allocate_index_id();
    
    // Create index entry
    IndexEntry entry;
    entry.clear();
    entry.index_id = index_id;
    entry.index_type = type;
    entry.status = IndexStatus::CREATING;
    entry.dimensions = config.dimensions;
    entry.max_elements = config.max_elements;
    entry.num_elements = 0;
    entry.create_time = static_cast<uint64_t>(std::time(nullptr));
    entry.modify_time = entry.create_time;
    std::strncpy(entry.name, name.c_str(), MAX_INDEX_NAME_LEN - 1);
    
    // Create index file
    std::string filename = storage_.make_index_filename(type, index_id, name);
    std::strncpy(entry.file_path, filename.c_str(), MAX_FILE_PATH_LEN - 1);
    
    // Store type-specific metadata
    if (type == IndexType::HNSW) {
        std::memcpy(entry.type_metadata, &config.hnsw, sizeof(HNSWConfig));
    } else if (type == IndexType::DISKANN) {
        std::memcpy(entry.type_metadata, &config.diskann, sizeof(DiskANNConfig));
    }
    
    // Open the index file (creates it)
    int fd = storage_.open_file(index_id, filename, true);
    if (fd < 0) {
        throw std::runtime_error("Failed to create index file: " + filename);
    }
    
    // Register index with BufferManager's multi-level translation array
    // This allocates a per-index translation array for hole-punching
    if (buffer_manager_) {
        // Initial capacity - array will grow dynamically as needed via mremap()
        // Start with a small capacity; ensureCapacity() will grow it automatically
        // This means collections can truly grow unbounded (limited only by virtual address space)
        uint64_t initial_pages;
        if (config.max_elements == 0) {
            // Unbounded collection - start small, will grow via mremap()
            initial_pages = 1024;  // ~4MB initial, grows automatically
        } else {
            // Fixed max_elements - estimate based on that
            initial_pages = (config.max_elements / 2) + 1024;
        }
        buffer_manager_->registerIndex(index_id, initial_pages, 0, fd);  // 0 = initial alloc count
    }
    
    // Add to catalog
    entries_.push_back(entry);
    name_to_index_id_[name] = index_id;
    header_.num_indexes++;
    
    // Mark as active
    entries_.back().status = IndexStatus::ACTIVE;
    
    // Persist catalog changes
    save_catalog_header();
    save_index_entry(entries_.back());
    
    CALIBY_LOG_INFO("IndexCatalog", "Created index: ", name, 
                   " (id=", index_id, ", type=", static_cast<int>(type), ")");
    
    return IndexHandle(this, index_id, fd, name, type, 
                      config.dimensions, config.max_elements);
}

IndexHandle IndexCatalog::create_hnsw_index(const std::string& name,
                                             uint32_t dimensions,
                                             uint64_t max_elements,
                                             size_t M,
                                             size_t ef_construction) {
    IndexConfig config;
    config.dimensions = dimensions;
    config.max_elements = max_elements;
    config.hnsw.M = M;
    config.hnsw.ef_construction = ef_construction;
    config.hnsw.enable_prefetch = false;
    
    return create_index(name, IndexType::HNSW, config);
}

IndexHandle IndexCatalog::create_diskann_index(const std::string& name,
                                                uint32_t dimensions,
                                                uint64_t max_elements,
                                                uint32_t R_max_degree,
                                                uint32_t L_build,
                                                float alpha) {
    IndexConfig config;
    config.dimensions = dimensions;
    config.max_elements = max_elements;
    config.diskann.R_max_degree = R_max_degree;
    config.diskann.L_build = L_build;
    config.diskann.alpha = alpha;
    
    return create_index(name, IndexType::DISKANN, config);
}

IndexHandle IndexCatalog::create_text_index(const std::string& name,
                                             const std::string& analyzer,
                                             const std::string& language,
                                             float k1,
                                             float b) {
    std::unique_lock lock(catalog_mutex_);
    
    if (!initialized_.load()) {
        throw std::runtime_error("Catalog not initialized");
    }
    
    // Check name doesn't exist
    if (name_to_index_id_.find(name) != name_to_index_id_.end()) {
        throw std::runtime_error("Index already exists: " + name);
    }
    
    // Validate name
    if (name.empty() || name.length() >= MAX_INDEX_NAME_LEN) {
        throw std::runtime_error("Invalid index name length");
    }
    
    // Allocate index ID
    uint32_t index_id = allocate_index_id();
    
    // Create index entry
    IndexEntry entry;
    entry.clear();
    entry.index_id = index_id;
    entry.index_type = IndexType::TEXT;
    entry.status = IndexStatus::CREATING;
    entry.dimensions = 0;
    entry.max_elements = 0;
    entry.num_elements = 0;
    entry.create_time = static_cast<uint64_t>(std::time(nullptr));
    entry.modify_time = entry.create_time;
    std::strncpy(entry.name, name.c_str(), MAX_INDEX_NAME_LEN - 1);
    
    // Create index file
    std::string filename = storage_.make_index_filename(IndexType::TEXT, index_id, name);
    std::strncpy(entry.file_path, filename.c_str(), MAX_FILE_PATH_LEN - 1);
    
    // Store text-specific metadata
    TextTypeMetadata text_meta;
    text_meta.initialize(analyzer, language, k1, b);
    std::memcpy(entry.type_metadata, &text_meta, sizeof(TextTypeMetadata));
    
    // Open the index file (creates it)
    int fd = storage_.open_file(index_id, filename, true);
    if (fd < 0) {
        throw std::runtime_error("Failed to create index file: " + filename);
    }
    
    // Register index with BufferManager
    if (buffer_manager_) {
        uint64_t initial_pages = 1024;  // Start small, grows automatically
        buffer_manager_->registerIndex(index_id, initial_pages, 0, fd);  // 0 = initial alloc count
    }
    
    // Add to catalog
    entries_.push_back(entry);
    name_to_index_id_[name] = index_id;
    header_.num_indexes++;
    
    // Mark as active
    entries_.back().status = IndexStatus::ACTIVE;
    
    // Persist catalog changes
    save_catalog_header();
    save_index_entry(entries_.back());
    
    CALIBY_LOG_INFO("IndexCatalog", "Created index: ", name, " (id=", index_id, ", type=TEXT)");
    
    return IndexHandle(this, index_id, fd, name, IndexType::TEXT, 0, 0);
}

IndexHandle IndexCatalog::create_btree_index(const std::string& name,
                                              const std::vector<std::string>& fields,
                                              bool unique) {
    std::unique_lock lock(catalog_mutex_);
    
    if (!initialized_.load()) {
        throw std::runtime_error("Catalog not initialized");
    }
    
    // Check name doesn't exist
    if (name_to_index_id_.find(name) != name_to_index_id_.end()) {
        throw std::runtime_error("Index already exists: " + name);
    }
    
    // Validate name
    if (name.empty() || name.length() >= MAX_INDEX_NAME_LEN) {
        throw std::runtime_error("Invalid index name length");
    }
    
    // Validate fields
    if (fields.empty() || fields.size() > BTreeTypeMetadata::MAX_FIELDS) {
        throw std::runtime_error("BTree index requires 1-" + 
                                std::to_string(BTreeTypeMetadata::MAX_FIELDS) + " fields");
    }
    
    // Allocate index ID
    uint32_t index_id = allocate_index_id();
    
    // Create index entry
    IndexEntry entry;
    entry.clear();
    entry.index_id = index_id;
    entry.index_type = IndexType::BTREE;
    entry.status = IndexStatus::CREATING;
    entry.dimensions = 0;
    entry.max_elements = 0;
    entry.num_elements = 0;
    entry.create_time = static_cast<uint64_t>(std::time(nullptr));
    entry.modify_time = entry.create_time;
    std::strncpy(entry.name, name.c_str(), MAX_INDEX_NAME_LEN - 1);
    
    // Create index file
    std::string filename = storage_.make_index_filename(IndexType::BTREE, index_id, name);
    std::strncpy(entry.file_path, filename.c_str(), MAX_FILE_PATH_LEN - 1);
    
    // Store btree-specific metadata
    BTreeTypeMetadata btree_meta;
    btree_meta.initialize(fields, unique);
    std::memcpy(entry.type_metadata, &btree_meta, sizeof(BTreeTypeMetadata));
    
    // Open the index file (creates it)
    int fd = storage_.open_file(index_id, filename, true);
    if (fd < 0) {
        throw std::runtime_error("Failed to create index file: " + filename);
    }
    
    // Register index with BufferManager
    if (buffer_manager_) {
        uint64_t initial_pages = 1024;  // Start small, grows automatically
        buffer_manager_->registerIndex(index_id, initial_pages, 0, fd);  // 0 = initial alloc count
    }
    
    // Add to catalog
    entries_.push_back(entry);
    name_to_index_id_[name] = index_id;
    header_.num_indexes++;
    
    // Mark as active
    entries_.back().status = IndexStatus::ACTIVE;
    
    // Persist catalog changes
    save_catalog_header();
    save_index_entry(entries_.back());
    
    CALIBY_LOG_INFO("IndexCatalog", "Created index: ", name, " (id=", index_id, ", type=BTREE)");
    
    return IndexHandle(this, index_id, fd, name, IndexType::BTREE, 0, 0);
}

IndexHandle IndexCatalog::open_index(const std::string& name) {
    std::shared_lock lock(catalog_mutex_);
    
    if (!initialized_.load()) {
        throw std::runtime_error("Catalog not initialized");
    }
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Index not found: " + name);
    }
    
    const IndexEntry* entry = find_entry_by_id(it->second);
    if (!entry || !entry->is_active()) {
        throw std::runtime_error("Index not active: " + name);
    }
    
    // Get the file descriptor (should already be open from initialize())
    int fd = storage_.get_fd(entry->index_id);
    if (fd < 0) {
        // If not open, open it now
        fd = const_cast<MultiFileStorage&>(storage_).open_file(
            entry->index_id, entry->file_path, false);
    }
    
    CALIBY_LOG_DEBUG("IndexCatalog", "Opened index: ", name, " (id=", entry->index_id, ")");
    
    return IndexHandle(const_cast<IndexCatalog*>(this), entry->index_id, fd, 
                      name, entry->index_type, entry->dimensions, entry->max_elements);
}

void IndexCatalog::drop_index(const std::string& name) {
    std::unique_lock lock(catalog_mutex_);
    
    if (!initialized_.load()) {
        throw std::runtime_error("Catalog not initialized");
    }
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Index not found: " + name);
    }
    
    uint32_t index_id = it->second;
    IndexEntry* entry = find_entry_by_id(index_id);
    if (!entry) {
        throw std::runtime_error("Index entry not found: " + name);
    }
    
    std::string filename = entry->file_path;
    
    // Mark as deleted
    entry->status = IndexStatus::DELETED;
    entry->modify_time = static_cast<uint64_t>(std::time(nullptr));
    save_index_entry(*entry);
    
    // Unregister index from BufferManager's multi-level translation array
    // This frees the per-index translation array
    if (buffer_manager_) {
        try {
            buffer_manager_->unregisterIndex(index_id);
        } catch (const std::exception& e) {
            // Log but don't throw - the index may not have been registered in Array2Level mode
            CALIBY_LOG_DEBUG("IndexCatalog", "Could not unregister index from buffer manager: ", e.what());
        }
    }
    
    // Close the file
    storage_.close_file(index_id);
    
    // Delete the file
    storage_.delete_file(filename);
    
    // Remove from in-memory structures
    name_to_index_id_.erase(name);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
        [index_id](const IndexEntry& e) { return e.index_id == index_id; }),
        entries_.end());
    
    header_.num_indexes--;
    save_catalog_header();
    
    CALIBY_LOG_INFO("IndexCatalog", "Dropped index: ", name);
}

bool IndexCatalog::index_exists(const std::string& name) const {
    std::shared_lock lock(catalog_mutex_);
    return name_to_index_id_.find(name) != name_to_index_id_.end();
}

std::vector<IndexInfo> IndexCatalog::list_indexes() const {
    std::shared_lock lock(catalog_mutex_);
    
    std::vector<IndexInfo> result;
    result.reserve(entries_.size());
    
    for (const auto& entry : entries_) {
        if (entry.is_active()) {
            IndexInfo info;
            info.index_id = entry.index_id;
            info.name = entry.name;
            info.type = entry.index_type;
            info.status = entry.status;
            info.dimensions = entry.dimensions;
            info.max_elements = entry.max_elements;
            info.num_elements = entry.num_elements;
            info.create_time = entry.create_time;
            info.modify_time = entry.modify_time;
            info.file_path = entry.file_path;
            result.push_back(info);
        }
    }
    
    return result;
}

IndexInfo IndexCatalog::get_index_info(const std::string& name) const {
    std::shared_lock lock(catalog_mutex_);
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Index not found: " + name);
    }
    
    const IndexEntry* entry = nullptr;
    for (const auto& e : entries_) {
        if (e.index_id == it->second) {
            entry = &e;
            break;
        }
    }
    
    if (!entry) {
        throw std::runtime_error("Index entry not found: " + name);
    }
    
    IndexInfo info;
    info.index_id = entry->index_id;
    info.name = entry->name;
    info.type = entry->index_type;
    info.status = entry->status;
    info.dimensions = entry->dimensions;
    info.max_elements = entry->max_elements;
    info.num_elements = entry->num_elements;
    info.create_time = entry->create_time;
    info.modify_time = entry->modify_time;
    info.file_path = entry->file_path;
    
    return info;
}

HNSWConfig IndexCatalog::get_hnsw_config(const std::string& name) const {
    std::shared_lock lock(catalog_mutex_);
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Index not found: " + name);
    }
    
    const IndexEntry* entry = nullptr;
    for (const auto& e : entries_) {
        if (e.index_id == it->second) {
            entry = &e;
            break;
        }
    }
    
    if (!entry) {
        throw std::runtime_error("Index entry not found: " + name);
    }
    
    if (entry->index_type != IndexType::HNSW) {
        throw std::runtime_error("Index is not HNSW type: " + name);
    }
    
    HNSWConfig config;
    std::memcpy(&config, entry->type_metadata, sizeof(HNSWConfig));
    return config;
}

TextTypeMetadata IndexCatalog::get_text_config(const std::string& name) const {
    std::shared_lock lock(catalog_mutex_);
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Index not found: " + name);
    }
    
    const IndexEntry* entry = nullptr;
    for (const auto& e : entries_) {
        if (e.index_id == it->second) {
            entry = &e;
            break;
        }
    }
    
    if (!entry) {
        throw std::runtime_error("Index entry not found: " + name);
    }
    
    if (entry->index_type != IndexType::TEXT) {
        throw std::runtime_error("Index is not TEXT type: " + name);
    }
    
    TextTypeMetadata config;
    std::memcpy(&config, entry->type_metadata, sizeof(TextTypeMetadata));
    return config;
}

BTreeTypeMetadata IndexCatalog::get_btree_config(const std::string& name) const {
    std::shared_lock lock(catalog_mutex_);
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Index not found: " + name);
    }
    
    const IndexEntry* entry = nullptr;
    for (const auto& e : entries_) {
        if (e.index_id == it->second) {
            entry = &e;
            break;
        }
    }
    
    if (!entry) {
        throw std::runtime_error("Index entry not found: " + name);
    }
    
    if (entry->index_type != IndexType::BTREE) {
        throw std::runtime_error("Index is not BTREE type: " + name);
    }
    
    BTreeTypeMetadata config;
    std::memcpy(&config, entry->type_metadata, sizeof(BTreeTypeMetadata));
    return config;
}

CollectionTypeMetadata IndexCatalog::get_collection_config(const std::string& name) const {
    std::shared_lock lock(catalog_mutex_);
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Collection not found: " + name);
    }
    
    const IndexEntry* entry = nullptr;
    for (const auto& e : entries_) {
        if (e.index_id == it->second) {
            entry = &e;
            break;
        }
    }
    
    if (!entry) {
        throw std::runtime_error("Collection entry not found: " + name);
    }
    
    if (entry->index_type != IndexType::COLLECTION) {
        throw std::runtime_error("Entry is not COLLECTION type: " + name);
    }
    
    CollectionTypeMetadata config;
    std::memcpy(&config, entry->type_metadata, sizeof(CollectionTypeMetadata));
    return config;
}

void IndexCatalog::update_collection_config(const std::string& name, const CollectionTypeMetadata& config) {
    std::unique_lock lock(catalog_mutex_);
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Collection not found: " + name);
    }
    
    IndexEntry* entry = nullptr;
    for (auto& e : entries_) {
        if (e.index_id == it->second) {
            entry = &e;
            break;
        }
    }
    
    if (!entry) {
        throw std::runtime_error("Collection entry not found: " + name);
    }
    
    if (entry->index_type != IndexType::COLLECTION) {
        throw std::runtime_error("Entry is not COLLECTION type: " + name);
    }
    
    std::memcpy(entry->type_metadata, &config, sizeof(CollectionTypeMetadata));
    entry->modify_time = static_cast<uint64_t>(std::time(nullptr));
    save_index_entry(*entry);
}

void IndexCatalog::update_text_config(const std::string& name, const TextTypeMetadata& config) {
    std::unique_lock lock(catalog_mutex_);
    
    CALIBY_LOG_DEBUG("IndexCatalog", "update_text_config for '", name, 
                     "': btree_slot=", config.btree_slot_id,
                     ", vocab=", config.vocab_size,
                     ", docs=", config.doc_count);
    
    auto it = name_to_index_id_.find(name);
    if (it == name_to_index_id_.end()) {
        throw std::runtime_error("Text index not found: " + name);
    }
    
    IndexEntry* entry = nullptr;
    for (auto& e : entries_) {
        if (e.index_id == it->second) {
            entry = &e;
            break;
        }
    }
    
    if (!entry) {
        throw std::runtime_error("Text index entry not found: " + name);
    }
    
    if (entry->index_type != IndexType::TEXT) {
        throw std::runtime_error("Entry is not TEXT type: " + name);
    }
    
    std::memcpy(entry->type_metadata, &config, sizeof(TextTypeMetadata));
    entry->modify_time = static_cast<uint64_t>(std::time(nullptr));
    save_index_entry(*entry);
}

void IndexCatalog::update_index_element_count(uint32_t index_id, uint64_t count) {
    std::unique_lock lock(catalog_mutex_);
    
    IndexEntry* entry = find_entry_by_id(index_id);
    if (entry) {
        entry->num_elements = count;
        entry->modify_time = static_cast<uint64_t>(std::time(nullptr));
        save_index_entry(*entry);
    }
}

void IndexCatalog::update_index_alloc_pages(uint32_t index_id, uint64_t alloc_pages) {
    std::unique_lock lock(catalog_mutex_);
    
    IndexEntry* entry = find_entry_by_id(index_id);
    if (entry) {
        entry->alloc_pages = alloc_pages;
        entry->modify_time = static_cast<uint64_t>(std::time(nullptr));
        save_index_entry(*entry);
        
        CALIBY_LOG_DEBUG("IndexCatalog", "Updated alloc_pages for index ", index_id, 
                         " to ", alloc_pages);
    }
}

uint64_t IndexCatalog::get_index_alloc_pages(uint32_t index_id) const {
    std::shared_lock lock(catalog_mutex_);
    
    // Need to cast away const for find_entry_by_id (which doesn't modify but isn't marked const)
    IndexEntry* entry = const_cast<IndexCatalog*>(this)->find_entry_by_id(index_id);
    if (entry) {
        return entry->alloc_pages;
    }
    return 0;
}

} // namespace caliby
