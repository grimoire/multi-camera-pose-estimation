#pragma once
#include "HCNetSDK.h"
#include "LinuxPlayM4.h"
#include <list>
#include <opencv2/opencv.hpp>

class HKIPcamera {
public:
  HKIPcamera();
  ~HKIPcamera();
  bool init(char *ip, char *usr, char *password, long port = 8000,
            long channel = 1, long streamtype = 0, long link_mode = 0,
            int device_id = 0, long buffer_size = 10);
  cv::Mat getframe();
  void release();
  void push_frame(const cv::Mat &frame);
  long get_buffer_size();
  bool is_buffer_full();

public:
  LONG lRealPlayHandle;
  int nPort_;
  HWND hWnd;
  LONG user_id_;
  pthread_mutex_t frame_list_mutex_;
  NET_DVR_DEVICEINFO_V30 struDeviceInfo_;
  long channel_;
  long streamtype_;
  long buffersize_;
  long linkmode_;
  long device_id_;

private:
  bool OpenCamera(char *ip, long port, char *usr, char *password);

private:
  std::list<cv::Mat> frame_list_;
};