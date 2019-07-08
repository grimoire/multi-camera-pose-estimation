#include "pipeline/pipe_process.h"
#include "pipeline/pipe_base.h"

class PipeProcessImpl : public PipeBase<int, SProcessOutput> {
public:
  PipeProcessImpl() {}
  ~PipeProcessImpl() {}
};

PipeProcess::PipeProcess()
    : _pipe_impl(nullptr), _multipose(nullptr), _pipe_tensor_merger(nullptr) {

  _pipe_impl = new PipeProcessImpl();
}

PipeProcess::~PipeProcess() {

  if (_pipe_impl != nullptr) {
    delete _pipe_impl;
  }
}

void PipeProcess::init_pipeline(MultiPose *multipose,
                                PipeTensorMerger *pipe_tensor_merger,
                                int out_size, const std::string &meta_info) {
  _multipose = multipose;
  _pipe_tensor_merger = pipe_tensor_merger;
  _pipe_impl->init_pipeline(0, out_size, meta_info);
}

std::shared_ptr<SProcessOutput> PipeProcess::get_data_out() {
  return _pipe_impl->getOutputInstance();
}

bool PipeProcess::write_out(const std::shared_ptr<SProcessOutput> &data) {
  return _pipe_impl->write_out(data);
}

bool PipeProcess::read_out(std::shared_ptr<SProcessOutput> &data) {
  return _pipe_impl->read_out(data);
}

bool PipeProcess::is_full_out() { return _pipe_impl->is_output_full(); }

void PipeProcess::clear_out() { _pipe_impl->clear_out(); }

void PipeProcess::start_thread() {
  _pipe_impl->start_pipeline(&PipeProcess::update, this);
}

bool PipeProcess::is_thread_stoped() {
  return _pipe_impl->is_pipeline_stoped();
}

void PipeProcess::stop_thread() { _pipe_impl->stop_pipeline(); }

void PipeProcess::update() {
  while (!is_thread_stoped()) {
    if (is_full_out()) {
      std::cout << "process " << _pipe_impl->get_meta_info() << " is full"
                << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
      continue;
    }
    if (_pipe_tensor_merger == nullptr) {
      std::cout << "please init merger first." << std::endl;
      continue;
    }
    auto input_data = _pipe_tensor_merger->get_data_out();
    while (!_pipe_tensor_merger->read_out(input_data)) {
      if (is_thread_stoped()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
    }

    _pipe_tensor_merger->clear_out();

    // auto start_time = std::chrono::steady_clock::now();
    auto result_tensor = _multipose->process(input_data->_merged_tensor);

    auto output_data = get_data_out();
    output_data->_image_list = input_data->_image_list;
    output_data->_meta_info_list = input_data->_meta_info_list;
    output_data->_meta_list = input_data->_meta_list;
    output_data->_result = result_tensor;

    // auto end_time = std::chrono::steady_clock::now();
    // auto elapse_milli =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // std::cout << "forward time " << elapse_milli.count()/1000. << " seconds." << std::endl;
    write_out(output_data);
  }
}
