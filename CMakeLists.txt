cmake_minimum_required(VERSION 3.16)
project(caliby VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(CALIBY_BUILD_TESTS "Build tests" OFF)
option(CALIBY_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CALIBY_BUILD_PYTHON "Build Python bindings" ON)
option(CALIBY_USE_OPENMP "Use OpenMP for parallelization" ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math -fno-omit-frame-pointer -g")

# Find required packages
find_package(Threads REQUIRED)

# OpenMP
if(CALIBY_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# libaio (Linux only)
if(UNIX AND NOT APPLE)
    find_library(AIO_LIBRARY aio)
    if(NOT AIO_LIBRARY)
        message(FATAL_ERROR "libaio not found. Install with: sudo apt-get install libaio-dev")
    endif()
endif()

# liburing (optional, for io_uring support)
find_library(URING_LIBRARY uring)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include/caliby
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/absl
)

# Find absl libraries (system-installed)
find_package(absl REQUIRED)
message(STATUS "Found absl")

# Set absl libraries to link
set(ABSL_LIBS 
    absl::hash 
    absl::raw_hash_set 
    absl::raw_logging_internal
)

# Source files for the core library
set(CALIBY_SOURCES
    src/calico.cpp
    src/hnsw.cpp
    src/diskann.cpp
    src/distance_diskann.cpp
    src/catalog.cpp
    src/ivfpq.cpp
    src/collection.cpp
    src/text_index.cpp
    src/btree_index.cpp
)

# Create static library
add_library(caliby_static STATIC ${CALIBY_SOURCES})
target_include_directories(caliby_static PUBLIC
    ${CMAKE_SOURCE_DIR}/include/caliby
    ${CMAKE_SOURCE_DIR}/third_party
)
target_link_libraries(caliby_static 
    PUBLIC Threads::Threads
    PRIVATE ${AIO_LIBRARY}
    PRIVATE ${ABSL_LIBS}
)

if(URING_LIBRARY)
    target_link_libraries(caliby_static PRIVATE ${URING_LIBRARY})
    target_compile_definitions(caliby_static PRIVATE CALIBY_HAS_URING)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(caliby_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# Create shared library (only when not building Python bindings)
if(NOT CALIBY_BUILD_PYTHON)
    add_library(libcaliby SHARED ${CALIBY_SOURCES})
    target_include_directories(libcaliby PUBLIC
        ${CMAKE_SOURCE_DIR}/include/caliby
        ${CMAKE_SOURCE_DIR}/third_party
    )
    target_link_libraries(libcaliby 
        PUBLIC Threads::Threads
        PRIVATE ${AIO_LIBRARY}
        PRIVATE ${ABSL_LIBS}
    )

    if(URING_LIBRARY)
        target_link_libraries(libcaliby PRIVATE ${URING_LIBRARY})
        target_compile_definitions(libcaliby PRIVATE CALIBY_HAS_URING)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(libcaliby PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Set library version
    set_target_properties(libcaliby PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        OUTPUT_NAME caliby
    )

    # Install targets
    install(TARGETS libcaliby caliby_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else()
    # Install only static library when building Python bindings
    install(TARGETS caliby_static
        ARCHIVE DESTINATION lib
    )
endif()

install(DIRECTORY include/caliby
    DESTINATION include
)

# Python bindings
if(CALIBY_BUILD_PYTHON)
    add_subdirectory(third_party/pybind11)
    
    # Create Python module
    pybind11_add_module(caliby MODULE src/bindings.cpp ${CALIBY_SOURCES})
    target_include_directories(caliby PRIVATE
        ${CMAKE_SOURCE_DIR}/include/caliby
        ${CMAKE_SOURCE_DIR}/third_party
    )
    target_link_libraries(caliby PRIVATE 
        Threads::Threads
        ${AIO_LIBRARY}
        ${ABSL_LIBS}
    )
    
    if(URING_LIBRARY)
        target_link_libraries(caliby PRIVATE ${URING_LIBRARY})
        target_compile_definitions(caliby PRIVATE CALIBY_HAS_URING)
    endif()
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(caliby PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Set proper output name and location for Python module
    set_target_properties(caliby PROPERTIES
        PREFIX ""
        OUTPUT_NAME "caliby"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()

# Tests
if(CALIBY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(CALIBY_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Caliby Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP:         ${CALIBY_USE_OPENMP}")
message(STATUS "  Python:         ${CALIBY_BUILD_PYTHON}")
message(STATUS "  Tests:          ${CALIBY_BUILD_TESTS}")
message(STATUS "  Benchmarks:     ${CALIBY_BUILD_BENCHMARKS}")
message(STATUS "  io_uring:       ${URING_LIBRARY}")
message(STATUS "")
