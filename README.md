# multi-camera-pose-estimation

realtime pose estimation with multi hikvision ipcamera
the pose estimation model is the c++ warp of centernet

https://github.com/xingyizhou/CenterNet

getting started
===============


Install
-------

build libtorch from the source(required)


download hikvision network sdk and player sdk

http://www1.hikvision.com/cn/download_more_403.html
http://www1.hikvision.com/cn/download_more_407.html


install nvidia cuda


config the CMakeLists.txt, set the path of necessary libs


copy all necessary librarys to ./lib or add the pathes to link_directories


make the project


Usage
=====

download the wrapped models here

https://pan.baidu.com/s/11XWNKf5nVxs4j9Kuer6Hsg

extract code: 7825

put the downloaded model in ./models

modify ./src/main.cpp, build and run

