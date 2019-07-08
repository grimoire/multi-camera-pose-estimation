#include "pipeline/pipe_postprocess.h"
#include "pipeline/pipe_base.h"
#include <chrono>
#include <thread>

class PipePostprocessImpl
    : public PipeBase<SPostprocessInput, SPostprocessOutput> {
public:
  PipePostprocessImpl() {}
  ~PipePostprocessImpl() {}
};

PipePostprocess::PipePostprocess()
    : _pipe_impl(nullptr), _multipose(nullptr), _out_pipe(nullptr) {
  _pipe_impl = new PipePostprocessImpl();
}

PipePostprocess::~PipePostprocess() {
  if (_pipe_impl != nullptr) {
    delete _pipe_impl;
  }
}

void PipePostprocess::init_pipeline(MultiPose *multipose, int in_size,
                                    int out_size,
                                    const std::string &meta_info) {
  _multipose = multipose;
  _pipe_impl->init_pipeline(in_size, out_size, meta_info);
}

void PipePostprocess::init_pipeline(MultiPose *multipose, int in_size,
                                    PipeOutput<SPostprocessOutput> *out_pipe,
                                    const std::string &meta_info) {
  _multipose = multipose;
  _out_pipe = out_pipe;
  _pipe_impl->init_pipeline(in_size, 0, meta_info);
}

std::shared_ptr<SPostprocessInput> PipePostprocess::get_data_in() {
  return _pipe_impl->getInputInstance();
}

bool PipePostprocess::write_in(const std::shared_ptr<SPostprocessInput> &data) {
  return _pipe_impl->write_in(data);
}

bool PipePostprocess::read_in(std::shared_ptr<SPostprocessInput> &data) {
  return _pipe_impl->read_in(data);
}

bool PipePostprocess::is_full_in() { return _pipe_impl->is_input_full(); }

void PipePostprocess::clear_in() { _pipe_impl->clear_in(); }

std::shared_ptr<SPostprocessOutput> PipePostprocess::get_data_out() {
  return _pipe_impl->getOutputInstance();
}

bool PipePostprocess::write_out(
    const std::shared_ptr<SPostprocessOutput> &data) {
  if (_out_pipe == nullptr) {
    return _pipe_impl->write_out(data);
  } else {
    return _out_pipe->write_out(data);
  }
}

bool PipePostprocess::read_out(std::shared_ptr<SPostprocessOutput> &data) {
  if (_out_pipe == nullptr) {
    return _pipe_impl->read_out(data);
  } else {
    return _out_pipe->read_out(data);
  }
}

bool PipePostprocess::is_full_out() {
  if (_out_pipe == nullptr) {
    return _pipe_impl->is_output_full();
  } else {
    return _out_pipe->is_output_full();
  }
}

void PipePostprocess::clear_out() {
  if (_out_pipe == nullptr) {
    _pipe_impl->clear_out();
  } else {
    _out_pipe->clear_out();
  }
}

void PipePostprocess::start_thread() {
  _pipe_impl->start_pipeline(&PipePostprocess::update, this);
}

bool PipePostprocess::is_thread_stoped() {
  return _pipe_impl->is_pipeline_stoped();
}

void PipePostprocess::stop_thread() { _pipe_impl->stop_pipeline(); }

void PipePostprocess::update() {
  while (!is_thread_stoped()) {
    if (is_full_out()) {
      std::cout << "postprocess " << _pipe_impl->get_meta_info() << " is full."
                << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
      continue;
    }

    auto input_data = get_data_in();
    while (!read_in(input_data)) {
      if (is_thread_stoped()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
    }


    // auto start_time = std::chrono::steady_clock::now();

    auto post_result =
        _multipose->postprocess(input_data->_result, input_data->_meta);
    auto output_data = get_data_out();
    output_data->_image = input_data->_image;
    output_data->_meta_info = input_data->_meta_info;
    output_data->_result = post_result.squeeze(0);

    write_out(output_data);

    // auto end_time = std::chrono::steady_clock::now();
    // auto elapse_milli = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     end_time - start_time);
    // std::cout << "postprocess time " << elapse_milli.count() / 1000.
    //           << " seconds." << std::endl;
  }
}