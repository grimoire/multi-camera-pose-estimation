#pragma once
#include "multi_pose/multi_pose.h"
#include <opencv2/core/core.hpp>
#include <pipeline/metainfo_map.h>
#include <string>
#include <torch/script.h>
#include <vector>

struct SMergerInput {
  cv::Mat _image;
  std::string _meta_info;
  at::Tensor _tensor_image;
  SPreprocessMetaParam _meta;
};

struct SMergerOutput {
  std::vector<cv::Mat> _image_list;
  std::vector<std::string> _meta_info_list;
  at::Tensor _merged_tensor;
  std::vector<SPreprocessMetaParam> _meta_list;
};

class PipeTensorMergerImpl;
class PipeTensorMerger {
public:
  PipeTensorMerger();
  ~PipeTensorMerger();

  void init_pipeline(MetaInfoMap *metainfo_map, int device_id, int batch_size, size_t in_size, size_t out_size,
                     const std::string &meta_info);

  std::shared_ptr<SMergerInput> get_data_in();
  virtual bool write_in(const std::shared_ptr<SMergerInput> &data);
  virtual bool read_in(std::shared_ptr<SMergerInput> &data);
  virtual bool is_full_in();
  virtual void clear_in();

  std::shared_ptr<SMergerOutput> get_data_out();
  virtual bool write_out(const std::shared_ptr<SMergerOutput> &data);
  virtual bool read_out(std::shared_ptr<SMergerOutput> &data);
  virtual bool is_full_out();
  virtual void clear_out();

  virtual void start_thread();
  virtual void stop_thread();
  virtual bool is_thread_stoped();

protected:
  virtual void update();

private:
  MetaInfoMap *_metainfo_map;
  PipeTensorMergerImpl *_pipe_impl;
  int _batch_size;
  int _device_id;
};