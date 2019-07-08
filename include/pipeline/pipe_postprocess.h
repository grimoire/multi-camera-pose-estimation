#pragma once
#include "multi_pose/multi_pose.h"
#include "pipeline/pipe_output.h"
#include <opencv2/core/core.hpp>
#include <string>
#include <torch/script.h>

struct SPostprocessInput {
  cv::Mat _image;
  std::string _meta_info;
  at::Tensor _result;
  SPreprocessMetaParam _meta;
};

struct SPostprocessOutput {
  cv::Mat _image;
  std::string _meta_info;
  at::Tensor _result;
};

class PipePostprocessImpl;
class PipePostprocess {
public:
  PipePostprocess();
  ~PipePostprocess();

  void init_pipeline(MultiPose *multipose, int in_size, int out_size,
                     const std::string &meta_info);

  void init_pipeline(MultiPose *multipose, int in_size, PipeOutput<SPostprocessOutput>* out_pipe,
                     const std::string &meta_info);

  std::shared_ptr<SPostprocessInput> get_data_in();
  virtual bool write_in(const std::shared_ptr<SPostprocessInput> &data);
  virtual bool read_in(std::shared_ptr<SPostprocessInput> &data);
  virtual bool is_full_in();
  virtual void clear_in();

  std::shared_ptr<SPostprocessOutput> get_data_out();
  virtual bool write_out(const std::shared_ptr<SPostprocessOutput> &data);
  virtual bool read_out(std::shared_ptr<SPostprocessOutput> &data);
  virtual bool is_full_out();
  virtual void clear_out();

  virtual void start_thread();
  virtual void stop_thread();
  virtual bool is_thread_stoped();

protected:
  virtual void update();

private:
  MultiPose *_multipose;
  PipePostprocessImpl *_pipe_impl;
  PipeOutput<SPostprocessOutput> *_out_pipe;
};