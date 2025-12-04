cmake_minimum_required(VERSION 3.14)
project(slam_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# System librealsense (librealsense2-dev) - optional
find_package(realsense2 QUIET)

# We want the Python extension to land inside slam/vio so that
# `import slam.vio.stereo_matching` works
# source_dir here is `cpp/`, Python package is ../slam/vio
set(PY_PKG_DIR "${CMAKE_SOURCE_DIR}/../slam/vio")

# ---- pybind11 module: stereo_matching ----
pybind11_add_module(stereo_matching stereo_matching.cpp)
target_link_libraries(stereo_matching PRIVATE ${OpenCV_LIBS})

set_target_properties(stereo_matching PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
)

# ---- pybind11 module: klt_tracker_cpp ----
pybind11_add_module(klt_tracker_cpp klt_tracker_cpp.cpp)
target_link_libraries(klt_tracker_cpp PRIVATE ${OpenCV_LIBS})

set_target_properties(klt_tracker_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
)

# ---- pybind11 module: rs_sdk (optional) ----
if(realsense2_FOUND)
    pybind11_add_module(rs_sdk rs_sdk.cpp)
    target_link_libraries(rs_sdk PRIVATE realsense2::realsense2 ${OpenCV_LIBS})

    set_target_properties(rs_sdk PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
    )
else()
    message(WARNING "realsense2 not found - rs_sdk module will not be built")
endif()
