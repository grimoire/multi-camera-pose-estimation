#pragma once
#include "multi_pose/multi_pose.h"
#include "pipeline/pipe_image.h"
#include "pipeline/pipe_tensormerger.h"
#include <memory>
#include <opencv2/core/core.hpp>
#include <string>
#include <torch/script.h>

class PipePreprocessImpl;

class PipePreprocess {
public:
  PipePreprocess();
  ~PipePreprocess();

  void initPipeline(PipeImage *pipe_image,
                    PipeTensorMerger *pipe_tensormerger,
                    MultiPose *multipose,
                    const std::string &meta_info);

  void setImageSize();

  virtual void start_thread();

  virtual void stop_thread();

  virtual bool is_thread_stoped();

protected:
  virtual void update();

protected:
  PipePreprocessImpl *_pipe_impl;

private:
  PipeImage *_pipe_image;
  PipeTensorMerger *_pipe_tensormerger;
  MultiPose *_multipose;
  cv::Size _image_size;
};