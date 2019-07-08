#pragma once
#include "pipeline/pipe_tensormerger.h"
#include <opencv2/core/core.hpp>
#include <string>
#include <torch/script.h>
#include <vector>

struct SProcessOutput {
  std::vector<cv::Mat> _image_list;
  std::vector<std::string> _meta_info_list;
  at::Tensor _result;
  std::vector<SPreprocessMetaParam> _meta_list;
};

class PipeProcessImpl;
class PipeProcess {
public:
  PipeProcess();
  ~PipeProcess();

  void init_pipeline(MultiPose *multipose, PipeTensorMerger *pipe_tensor_merger,
                     int out_size, const std::string &meta_info);

  std::shared_ptr<SProcessOutput> get_data_out();
  virtual bool write_out(const std::shared_ptr<SProcessOutput> &data);
  virtual bool read_out(std::shared_ptr<SProcessOutput> &data);
  virtual bool is_full_out();
  virtual void clear_out();

  virtual void start_thread();
  virtual void stop_thread();
  virtual bool is_thread_stoped();

protected:
  virtual void update();

private:
  MultiPose *_multipose;
  PipeTensorMerger *_pipe_tensor_merger;
  PipeProcessImpl *_pipe_impl;
};