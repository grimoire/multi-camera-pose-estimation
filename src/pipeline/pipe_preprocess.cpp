#include "pipeline/pipe_preprocess.h"
#include "pipeline/pipe_base.h"
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>

class PipePreprocessImpl : public PipeBase<int, int> {
public:
  PipePreprocessImpl() {}
  ~PipePreprocessImpl() {}
};

PipePreprocess::PipePreprocess()
    : _pipe_image(nullptr), _pipe_tensormerger(nullptr), _multipose(nullptr) {
  _image_size = cv::Size(0, 0);
  _pipe_impl = new PipePreprocessImpl();
}

PipePreprocess::~PipePreprocess() {
  if (_pipe_impl != nullptr) {
    delete _pipe_impl;
  }
}

void PipePreprocess::initPipeline(PipeImage *pipe_image,
                                  PipeTensorMerger *pipe_tensormerger,
                                  MultiPose *multipose,
                                  const std::string &meta_info) {
  _pipe_image = pipe_image;
  _pipe_tensormerger = pipe_tensormerger;
  _multipose = multipose;

  _pipe_impl->init_pipeline(0, 0, meta_info);
}

void PipePreprocess::start_thread() {
  _pipe_impl->start_pipeline(&PipePreprocess::update, this);
}

bool PipePreprocess::is_thread_stoped() {
  return _pipe_impl->is_pipeline_stoped();
}

void PipePreprocess::stop_thread() { _pipe_impl->stop_pipeline(); }

void PipePreprocess::update() {
  if (_pipe_image == nullptr || _pipe_tensormerger == nullptr ||
      _multipose == nullptr) {
    std::cout << "please init preprocess pipeline first" << std::endl;
  }
  while (!is_thread_stoped()) {
    // auto start_time = std::chrono::steady_clock::now();
    if (_pipe_tensormerger->is_full_in()) {
      // std::cout <<" merger input full."<<std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
      continue;
    }

    auto image_data = _pipe_image->get_data_inst();
    while (!(_pipe_image->read_out(image_data))) {
      if (is_thread_stoped()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
    }
    _pipe_image->clear_out();

    SPreprocessMetaParam meta;
    cv::Mat image = image_data->_image;
    if (_image_size.area() > 0) {
      cv::resize(image, image, _image_size);
    }
    auto image_tensor = _multipose->preprocess(image, meta);

    auto merger_input = _pipe_tensormerger->get_data_in();
    merger_input->_image = image_data->_image;
    merger_input->_meta_info = image_data->_meta_info;
    merger_input->_tensor_image = image_tensor;
    merger_input->_meta = meta;
    _pipe_tensormerger->write_in(merger_input);
  }
}