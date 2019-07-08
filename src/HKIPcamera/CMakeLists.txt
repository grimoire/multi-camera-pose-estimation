cmake_minimum_required(VERSION 2.8.11)

# npp
add_subdirectory(npp_cuda)
set(EXTRA_LIBS ${EXTRA_LIBS} npp_cuda)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-rpath=\"./\" -Wl,-rpath=\"./lib/\"")
add_definitions(-std=c++11)
add_compile_options(-fPIC)

include_directories(${HIKVISION_INCLUDE} ${PYTHON35_INCLUDE})
link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(${PROJECT_SOURCE_DIR}/lib/HCNetSDKCom)

file(GLOB HKIPCam_CODE *.h *.cpp)

set(HKIPCam_LIBS 	AudioRender
					HCCore
					PlayCtrl
					SuperRender
					crypto
					hcnetsdk
					hpr
					opencv_world
					ssl
					HCAlarm
					HCCoreDevCfg
					HCDisplay
					HCGeneralCfgMgr
					HCIndustry
					HCPlayBack
					HCPreview
					HCVoiceTalk
					StreamTransClient
					SystemTransform
					analyzedata
					iconv2
                     ${EXTRA_LIBS} 
                     opencv_world
                    )


add_library(HKIPcamera SHARED ${HKIPCam_CODE})

target_link_libraries(HKIPcamera PRIVATE ${HKIPCam_LIBS})
# file(COPY ${PROJECT_SOURCE_DIR}/lib DESTINATION ${CMAKE_BINARY_DIR}/lib)