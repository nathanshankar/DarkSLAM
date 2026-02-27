cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3)

# Get cuDNN version
message('${CUDNN_INCLUDE_PATH}')

# If not defined from command line, default to system include path
if(NOT DEFINED CUDNN_INCLUDE_PATH)
    set(CUDNN_INCLUDE_PATH "/usr/include")
endif()

message(STATUS "CUDNN_INCLUDE_PATH = ${CUDNN_INCLUDE_PATH}")

if(EXISTS "${CUDNN_INCLUDE_PATH}/cudnn_version.h")
    file(READ "${CUDNN_INCLUDE_PATH}/cudnn_version.h" CUDNN_HEADER_CONTENTS)
elseif(EXISTS "${CUDNN_INCLUDE_PATH}/cudnn.h")
    file(READ "${CUDNN_INCLUDE_PATH}/cudnn.h" CUDNN_HEADER_CONTENTS)
else()
    message(WARNING "cuDNN headers not found in ${CUDNN_INCLUDE_PATH}, continuing without cuDNN")
endif()


### LIBTORCH
#set(Torch_DIR "/home/ubuntu/pytorch_master/pytorch/torch/share/cmake/Torch")  # libtorch
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch/")

set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
message(STATUS "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
###

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -O3")  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")  
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")  
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native") 

# Check C++17 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# cmake_policy(SET CMP0146 NEW) # mine
find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
# ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW3/src
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${TORCH_INCLUDE_DIRS}
)
# if(NOT "${OpenCV_INCLUDE_DIRS}" IN_LIST CMAKE_INCLUDE_PATH)
#     include_directories(${OpenCV_INCLUDE_DIRS})
# endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
# src/ORBextractor.cc
# src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/inference.cpp
src/SuperPointExtractor.cc
src/SuperPointMatcher.cc
src/GTViewer.cc
# src/SuperPoint.cc
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
# include/ORBextractor.h
# include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/inference.h
include/SuperPointExtractor.h
include/SuperPointMatcher.h
# include/SuperPoint.h
include/GTViewer.h
)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBow3/build/src/libDBoW3.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${TORCH_LIBRARIES}
-lboost_serialization
-lcrypto
)



# # Build examples

# #Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)


add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti 
                     ${PROJECT_NAME}
                     ${TORCH_LIBRARIES} )

add_executable(convert_voc
        Examples/Monocular/convert_voc.cpp)
target_link_libraries(convert_voc 
                     ${PROJECT_NAME})

add_executable(tune_parameters
        Examples/Monocular/tune_parameters.cc)
target_link_libraries(tune_parameters
                     ${PROJECT_NAME}
                     ${TORCH_LIBRARIES} )

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti 
                     ${PROJECT_NAME}
                     ${TORCH_LIBRARIES} )


add_executable(keypoints_per_frame
        Examples/Stereo/keypoints_per_frame.cc)
target_link_libraries(keypoints_per_frame 
                     ${PROJECT_NAME}
                     ${TORCH_LIBRARIES} )


#RGBD examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})