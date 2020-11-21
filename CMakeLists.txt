cmake_minimum_required (VERSION 2.8.8)
project (CudaHardDecodeRtsp_Prj)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")
#
#aux_source_directory(./src DIR_SRCS)
set(SOURCE_FILES ./src/MainDemo.cpp ./src/NvDecode.cpp ./src/FrameQueue.cpp 
./src/cudaModuleMgr.cpp ./src/cudaProcessFrame.cpp ./src/dynlink_cuda.cpp ./src/dynlink_nvcuvid.cpp
)

link_directories(
        ./lib/3rd
	./lib/common
        ./lib/ffmpeg
	./lib/opencv
)  # 
include_directories(
        ./include
	./include/common/inc
	./include/ffmpeg
	./include/ffnvcodec
	./include/opencv 
)

add_executable(hdDemo ${SOURCE_FILES})

target_link_libraries(hdDemo cudart avcodec avformat avutil opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc dl-2.27 pthread-2.27)

