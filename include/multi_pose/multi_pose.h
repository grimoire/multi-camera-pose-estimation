#pragma once
#include <opencv2/core/core.hpp>
#include <string>
#include <torch/script.h>

class MultiPoseImpl;

struct SMultiPoseParam {
  int _process_size;
  int _device_id;

  SMultiPoseParam() : _process_size(384), _device_id(0) {}
};

struct SPreprocessMetaParam {
  float _center[2];
  float _scale;
  float _out_height;
  float _out_width;
};

class MultiPose {
private:
  MultiPoseImpl *_impl;

public:
  MultiPose();
  ~MultiPose();

  bool load_model(const std::string &model_path);

  void set_param(SMultiPoseParam &param);

  SMultiPoseParam &get_param();

  at::Tensor preprocess(const cv::Mat &image, SPreprocessMetaParam &meta);

  at::Tensor process(const at::Tensor &tensor, int K = 100);
  
  at::Tensor postprocess(const at::Tensor &forward_result,
                         const SPreprocessMetaParam &meta);
};