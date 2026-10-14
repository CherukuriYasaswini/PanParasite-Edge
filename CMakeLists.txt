cmake_minimum_required(VERSION 3.20)

project(panparasite_edge VERSION 0.1.0 LANGUAGES C CXX)

# ------------------------------------------------------------------
# Language standard (OPENCODE.md Rule 1: C++20)
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for edge performance if not specified.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------
# Platform detection + simulation fallbacks for O_DIRECT / mlock.
# Linux ARM64 (Raspberry Pi 5) is the deployment target.
# Windows/x86_64 is a simulation fallback (no real O_DIRECT/mlock).
# ------------------------------------------------------------------
if(WIN32)
  # Simulation fallbacks: O_DIRECT is a no-op, mlock -> VirtualLock.
  add_compile_definitions(
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    PANPARASITE_SIMULATE_DIRECT_IO=1
    PANPARASITE_SIMULATE_MLOCK=1
  )
else()
  # Needed for O_DIRECT on Linux/glibc.
  add_compile_definitions(_GNU_SOURCE=1)
endif()

# ------------------------------------------------------------------
# Dependencies: OpenCV, SQLite3, OpenSSL, Threads(pthread)
# ------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# SQLite3: CMake >=3.14 ships FindSQLite3 (target SQLite::SQLite3).
# Provide a portable fallback via pkg-config / manual search so that
# Windows + vcpkg and minimal Pi images both work.
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND AND NOT TARGET SQLite::SQLite3)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(SQLITE3_PC QUIET sqlite3)
  endif()
  find_path(SQLite3_INCLUDE_DIR NAMES sqlite3.h
    HINTS ${SQLITE3_PC_INCLUDEDIR} ENV SQLITE3_INCLUDE_DIR)
  find_library(SQLite3_LIBRARY NAMES sqlite3 sqlite3_i
    HINTS ${SQLITE3_PC_LIBDIR} ENV SQLITE3_LIBRARY_DIR)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SQLite3
    REQUIRED_VARS SQLite3_LIBRARY SQLite3_INCLUDE_DIR)
  if(SQLite3_FOUND AND NOT TARGET SQLite::SQLite3)
    add_library(SQLite::SQLite3 UNKNOWN IMPORTED)
    set_target_properties(SQLite::SQLite3 PROPERTIES
      IMPORTED_LOCATION "${SQLite3_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIR}")
  endif()
endif()
if(NOT SQLite3_FOUND AND NOT TARGET SQLite::SQLite3)
  message(FATAL_ERROR
    "SQLite3 not found. Install libsqlite3-dev (Linux) or sqlite3 via vcpkg (Windows).")
endif()

# ------------------------------------------------------------------
# Core static library: Phases 1-5 (arena → tracker) + Phase 6 (Stage 7
# persistence, Stage 8 telemetry/UI, end-to-end pipeline orchestrator).
# ------------------------------------------------------------------
add_library(panparasite_edge STATIC
  src/memory_arena.cpp
  src/preprocessor.cpp
  src/topology_engine.cpp
  src/active_perception.cpp
  src/inference_tracker.cpp
  src/persistence.cpp
  src/telemetry_ui.cpp
  src/pipeline.cpp
)

target_include_directories(panparasite_edge
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(panparasite_edge
  PUBLIC
    opencv_core
    opencv_imgproc
    opencv_dnn
    OpenSSL::Crypto
    SQLite::SQLite3
    Threads::Threads
)

# Windows system libs for VirtualLock (_aligned_malloc lives in CRT,
# VirtualLock/VirtualUnlock live in kernel32 which is linked by default).
if(WIN32)
  target_link_libraries(panparasite_edge PUBLIC ws2_32)
endif()

target_compile_features(panparasite_edge PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(panparasite_edge PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(panparasite_edge PRIVATE -Wall -Wextra -Wpedantic)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(panparasite_edge PRIVATE -O3)
  endif()
  # Pi 5 (Cortex-A76, ARMv8.2): enable CRC crypto extensions used by
  # Stage 7 CRC32-C when building on ARM64 Linux. No-op on x86_64.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    target_compile_options(panparasite_edge PRIVATE -march=armv8-a+crc+crypto)
  endif()
endif()

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
enable_testing()

add_executable(test_arena tests/test_arena.cpp)

target_link_libraries(test_arena
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

# Test binary also needs C++20 (for sized-delete overloads, etc.).
target_compile_features(test_arena PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_arena PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_arena PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME arena_zero_alloc COMMAND test_arena)

add_executable(test_preprocessor tests/test_preprocessor.cpp)

target_link_libraries(test_preprocessor
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_preprocessor PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_preprocessor PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_preprocessor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME preprocessor_stage1_stage2 COMMAND test_preprocessor)

add_executable(test_topology tests/test_topology.cpp)

target_link_libraries(test_topology
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_topology PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_topology PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_topology PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME topology_stage3 COMMAND test_topology)

add_executable(test_active_perception tests/test_active_perception.cpp)

target_link_libraries(test_active_perception
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_active_perception PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_active_perception PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_active_perception PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME active_perception_stage4 COMMAND test_active_perception)

add_executable(test_inference_tracker tests/test_inference_tracker.cpp)

target_link_libraries(test_inference_tracker
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_inference_tracker PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_inference_tracker PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_inference_tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME inference_tracker_stage5_6 COMMAND test_inference_tracker)

add_executable(test_persistence tests/test_persistence.cpp)

target_link_libraries(test_persistence
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_persistence PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_persistence PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_persistence PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME persistence_stage7 COMMAND test_persistence)

add_executable(test_telemetry tests/test_telemetry.cpp)

target_link_libraries(test_telemetry
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_telemetry PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_telemetry PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_telemetry PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME telemetry_stage8 COMMAND test_telemetry)

add_executable(test_pipeline tests/test_pipeline.cpp)

target_link_libraries(test_pipeline
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_pipeline PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_pipeline PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_pipeline PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME pipeline_end_to_end COMMAND test_pipeline)

add_executable(test_frontend tests/test_frontend.cpp)

target_link_libraries(test_frontend
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

target_compile_features(test_frontend PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(test_frontend PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(test_frontend PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME frontend_micropara COMMAND test_frontend)

# Main CLI benchmark executable. The logical target is named
# panparasite_cli (the panparasite_edge name is taken by the library);
# OUTPUT_NAME gives the produced binary the required panparasite_edge name.
add_executable(panparasite_cli src/main.cpp)

target_link_libraries(panparasite_cli
  PRIVATE
    panparasite_edge
    opencv_core
    Threads::Threads
)

set_target_properties(panparasite_cli PROPERTIES OUTPUT_NAME panparasite_edge)

target_compile_features(panparasite_cli PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(panparasite_cli PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(panparasite_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Convenience: build-time summary.
message(STATUS "PanParasite-Edge Phase 6: C++${CMAKE_CXX_STANDARD} | "
  "Build=${CMAKE_BUILD_TYPE} | System=${CMAKE_SYSTEM_NAME} "
  "| Processor=${CMAKE_SYSTEM_PROCESSOR}")
