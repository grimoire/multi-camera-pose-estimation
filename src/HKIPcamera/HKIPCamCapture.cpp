#include "HKIPcamera/HKIPCamCapture.h"
#include "HKIPcamera.h"
#include <memory>

HKIPCamCapture::HKIPCamCapture() : _hkipc(nullptr), _is_opened(false) {
  _hkipc = new HKIPcamera();
}

HKIPCamCapture::~HKIPCamCapture() {
  release();
  if (_hkipc) {
    delete _hkipc;
    _hkipc = nullptr;
  }
}

bool HKIPCamCapture::open() {
  if (isOpened()) {
    release();
  }
  const int kMAX_STR_SIZE = 50;
  char ip_str[kMAX_STR_SIZE];
  char username_str[kMAX_STR_SIZE];
  char password_str[kMAX_STR_SIZE];
  strcpy(ip_str, _conn_param.ip.c_str());
  strcpy(username_str, _conn_param.username.c_str());
  strcpy(password_str, _conn_param.password.c_str());
  _is_opened = _hkipc->init(ip_str, username_str, password_str,
                            _conn_param.port, _conn_param.channel,
                            _conn_param.streamtype, _conn_param.link_mode,
                            _conn_param.device_id, _conn_param.buffer_size);
  return isOpened();
}

bool HKIPCamCapture::read(cv::Mat &image) {
  if (!isOpened())
    return false;
  image = _hkipc->getframe();
  return true;
}

HKIPCamCapture &HKIPCamCapture::operator>>(cv::Mat &image) {
  if (isOpened()) {
    read(image);
  }
  return *this;
}

void HKIPCamCapture::release() {
  if (!_hkipc || !isOpened()) {
    return;
  }
  _hkipc->release();
  _is_opened = false;
}