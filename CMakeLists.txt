cmake_minimum_required(VERSION 3.16)
project(DarkSLAM LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(COMPILEDWITHC11)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV 4.4 REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(BUILD_UTILS OFF CACHE BOOL "Build DBoW3 utilities" FORCE)
add_subdirectory(Thirdparty/DBow3)
add_subdirectory(Thirdparty/g2o)

add_library(ORB_SLAM3 SHARED
    src/System.cc src/Tracking.cc src/LocalMapping.cc src/LoopClosing.cc
    src/FrameDrawer.cc src/Converter.cc src/MapPoint.cc src/KeyFrame.cc
    src/Atlas.cc src/Map.cc src/Optimizer.cc src/Frame.cc
    src/KeyFrameDatabase.cc src/Sim3Solver.cc src/ImuTypes.cc src/G2oTypes.cc
    src/CameraModels/Pinhole.cpp src/CameraModels/KannalaBrandt8.cpp
    src/OptimizableTypes.cpp src/MLPnPsolver.cpp src/GeometricTools.cc
    src/TwoViewReconstruction.cc src/Config.cc src/Settings.cc
    src/SuperPointExtractor.cc src/SuperPointMatcher.cc)
target_include_directories(ORB_SLAM3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CameraModels
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/Sophus
    ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(ORB_SLAM3 PUBLIC
    ${OpenCV_LIBS} Eigen3::Eigen DBoW3 g2o ${TORCH_LIBRARIES}
    Boost::serialization OpenSSL::Crypto Threads::Threads)
set_target_properties(ORB_SLAM3 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_executable(rgbd_tum_headless Examples/RGB-D/rgbd_tum_headless.cc)
target_link_libraries(rgbd_tum_headless PRIVATE ORB_SLAM3)
set_target_properties(rgbd_tum_headless PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
