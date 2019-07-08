#pragma once
#include "pipeline/pipe_image.h"
#include <string>

class HKIPCamCapture;
class PipeImageHKIPCam : public PipeImage {
public:
  PipeImageHKIPCam();
  ~PipeImageHKIPCam();

  bool init_camera(const std::string &ip, const std::string &usr,
                   const std::string &password, long port = 8000,
                   long channel = 1, long streamtype = 0);

  bool set_link_mode(long link_mode);

  bool set_cuda_device_id(int device_id);

  bool set_ipcam_buffer_size(long buffer_size);

protected:
  virtual void update();

private:
  HKIPCamCapture *_hkipc;
};