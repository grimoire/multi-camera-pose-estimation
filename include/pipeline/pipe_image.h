#pragma once
#include <memory>
#include <opencv2/core/core.hpp>
#include <string>

struct SImageData {
  cv::Mat _image;
  std::string _meta_info;
};

class PipeImageImpl;
class PipeImage {
public:
  PipeImage();
  ~PipeImage();
  void init_pipeline(size_t out_size, const std::string &meta_info);

  std::shared_ptr<SImageData> get_data_inst();

  virtual bool write_out(const std::shared_ptr<SImageData> &data);
  virtual bool read_out(std::shared_ptr<SImageData> &data);
  virtual bool is_full_out();
  virtual void clear_out();

  virtual void start_thread();

  virtual void stop_thread();
  
  virtual bool is_thread_stoped();

  void set_frame_rate(int frame_rate) { _frame_rate = frame_rate; }

  int get_frame_rate() { return _frame_rate; }

  std::string get_meta_info();
protected:
  virtual void update() = 0;

protected:
  PipeImageImpl *_pipe_impl;
  int _frame_rate;
};