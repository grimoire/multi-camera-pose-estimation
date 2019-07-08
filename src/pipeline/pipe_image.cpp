#include "pipeline/pipe_image.h"
#include "pipeline/pipe_base.h"

class PipeImageImpl : public PipeBase<int, SImageData> {
public:
  PipeImageImpl() {}
  ~PipeImageImpl() {}
};

PipeImage::PipeImage() : _frame_rate(25), _pipe_impl(nullptr) {
  _pipe_impl = new PipeImageImpl();
}

PipeImage::~PipeImage() {
  if (_pipe_impl != nullptr) {
    delete _pipe_impl;
  }
}

void PipeImage::init_pipeline(size_t out_size, const std::string &meta_info) {
  _pipe_impl->init_pipeline(0, out_size, meta_info);
}

std::shared_ptr<SImageData> PipeImage::get_data_inst() {
  return _pipe_impl->getOutputInstance();
}

bool PipeImage::write_out(const std::shared_ptr<SImageData> &data) {
  return _pipe_impl->write_out(data);
}

bool PipeImage::read_out(std::shared_ptr<SImageData> &data) {
  return _pipe_impl->read_out(data);
}

bool PipeImage::is_full_out(){
  return _pipe_impl->is_output_full();
}

void PipeImage::clear_out() { _pipe_impl->clear_out(); }

void PipeImage::start_thread() {
  _pipe_impl->start_pipeline(&PipeImage::update, this);
}

bool PipeImage::is_thread_stoped() { return _pipe_impl->is_pipeline_stoped(); }

void PipeImage::stop_thread() { _pipe_impl->stop_pipeline(); }

void PipeImage::update() {}

std::string PipeImage::get_meta_info(){
  return _pipe_impl->get_meta_info();
}