cmake_minimum_required(VERSION 3.0.2)
project(hba)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x -std=c++17 -fexceptions")

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(yaml-cpp REQUIRED)

include_directories(
  include

  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

# Hierarchical Bundle Adjustment
add_executable(hba source/hba.cpp)
target_link_libraries(hba ${PCL_LIBRARIES} gtsam yaml-cpp)

add_executable(visualize source/visualize.cpp)
target_link_libraries(visualize ${PCL_LIBRARIES} yaml-cpp)

add_executable(visualizeSeq source/visualizeSeq.cpp)
target_link_libraries(visualizeSeq ${PCL_LIBRARIES} yaml-cpp)

add_executable(savepcd source/savepcd.cpp)
target_link_libraries(savepcd ${PCL_LIBRARIES} yaml-cpp)
