#include "pipeline/pipe_image_hkipcam.h"
#include "HKIPcamera/HKIPCamCapture.h"
#include <chrono>
#include <iostream>
#include <thread>

PipeImageHKIPCam::PipeImageHKIPCam() : _hkipc(nullptr) {
  _hkipc = new HKIPCamCapture();
}

PipeImageHKIPCam::~PipeImageHKIPCam() {
  if (_hkipc) {
    delete _hkipc;
    _hkipc = nullptr;
  }
}

bool PipeImageHKIPCam::init_camera(const std::string &ip,
                                   const std::string &usr,
                                   const std::string &password, long port,
                                   long channel, long streamtype) {
  if (!_hkipc)
    return false;

  auto conn_param = _hkipc->getConnectParam();
  conn_param.ip = ip;
  conn_param.username = usr;
  conn_param.password = password;
  conn_param.port = port;
  conn_param.channel = channel;
  conn_param.streamtype = streamtype;
  conn_param.link_mode = 1;
  conn_param.device_id = -1;
  conn_param.buffer_size = 60;
  _hkipc->setConnectParam(conn_param);
  return true;
}

bool PipeImageHKIPCam::set_link_mode(long link_mode) {
  if (!_hkipc)
    return false;

  auto conn_param = _hkipc->getConnectParam();
  conn_param.link_mode = link_mode;
  return true;
}

bool PipeImageHKIPCam::set_cuda_device_id(int device_id) {
  if (!_hkipc)
    return false;

  auto conn_param = _hkipc->getConnectParam();
  conn_param.device_id = device_id;
  return true;
}

bool PipeImageHKIPCam::set_ipcam_buffer_size(long buffer_size) {
  if (!_hkipc)
    return false;

  auto conn_param = _hkipc->getConnectParam();
  conn_param.buffer_size = buffer_size;
  return true;
}

void PipeImageHKIPCam::update() {
  bool success = _hkipc->open();
  if (!success)
    return;

  int max_frame_time = int(1000 / get_frame_rate());
  while (!is_thread_stoped()) {
    if (is_full_out()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(max_frame_time));
      continue;
    }
    auto start_time = std::chrono::steady_clock::now();

    auto data = get_data_inst();
    bool read_success = _hkipc->read(data->_image);
    data->_meta_info = get_meta_info();
    if (!read_success) {
      break;
    }
    auto end_time = std::chrono::steady_clock::now();
    auto elapse_milli = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    if (elapse_milli.count() < max_frame_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(max_frame_time) -
                                  elapse_milli);
    } else {
      // std::cout << elapse_milli.count() << std::endl;
    }
    write_out(data);
  }
  _hkipc->release();
}