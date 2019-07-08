#pragma once
#include <opencv2/core/core.hpp>
#include <string>

class HKIPcamera;

struct SIPCam_Connect_Param {
  std::string ip;
  std::string username;
  std::string password;
  long port = 8000;
  long channel = 1;
  long streamtype = 0;
  long link_mode = 0;
  int device_id = 0;
  long buffer_size = 10;
};

class HKIPCamCapture {
public:
  HKIPCamCapture();
  ~HKIPCamCapture();

  void setConnectParam(const SIPCam_Connect_Param &conn_param) {
    _conn_param = conn_param;
  }

  SIPCam_Connect_Param &getConnectParam() { return _conn_param; }
  const SIPCam_Connect_Param &getConnectParam() const{ return _conn_param; }

  bool open();
  bool isOpened() const { return _is_opened; }
  bool read(cv::Mat &image);
  HKIPCamCapture &operator>>(cv::Mat &image);
  void release();

private:
  SIPCam_Connect_Param _conn_param;
  HKIPcamera *_hkipc;
  bool _is_opened;
};