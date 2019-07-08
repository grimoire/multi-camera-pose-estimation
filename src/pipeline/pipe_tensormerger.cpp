#include "pipeline/pipe_tensormerger.h"
#include "pipeline/pipe_base.h"
#include <chrono>
#include <thread>

class PipeTensorMergerImpl : public PipeBase<SMergerInput, SMergerOutput> {
public:
  PipeTensorMergerImpl() {}
  ~PipeTensorMergerImpl() {}
};

PipeTensorMerger::PipeTensorMerger() : _batch_size(8), _pipe_impl(nullptr) {
  _pipe_impl = new PipeTensorMergerImpl();
}

PipeTensorMerger::~PipeTensorMerger() {
  if (_pipe_impl != nullptr) {
    delete _pipe_impl;
  }
}

void PipeTensorMerger::init_pipeline(MetaInfoMap *metainfo_map, int device_id,
                                     int batch_size, size_t in_size,
                                     size_t out_size,
                                     const std::string &meta_info) {
  _device_id = device_id;
  _metainfo_map = metainfo_map;
  _batch_size = batch_size;
  _pipe_impl->init_pipeline(in_size, out_size, meta_info);
}

std::shared_ptr<SMergerInput> PipeTensorMerger::get_data_in() {
  return _pipe_impl->getInputInstance();
}

bool PipeTensorMerger::write_in(const std::shared_ptr<SMergerInput> &data) {
  return _pipe_impl->write_in(data);
}

bool PipeTensorMerger::read_in(std::shared_ptr<SMergerInput> &data) {
  return _pipe_impl->read_in(data);
}

bool PipeTensorMerger::is_full_in() { return _pipe_impl->is_input_full(); }

void PipeTensorMerger::clear_in() { _pipe_impl->clear_in(); }

std::shared_ptr<SMergerOutput> PipeTensorMerger::get_data_out() {
  return _pipe_impl->getOutputInstance();
}

bool PipeTensorMerger::write_out(const std::shared_ptr<SMergerOutput> &data) {
  return _pipe_impl->write_out(data);
}

bool PipeTensorMerger::read_out(std::shared_ptr<SMergerOutput> &data) {
  return _pipe_impl->read_out(data);
}

bool PipeTensorMerger::is_full_out() { return _pipe_impl->is_output_full(); }

void PipeTensorMerger::clear_out() { _pipe_impl->clear_out(); }

void PipeTensorMerger::start_thread() {
  _pipe_impl->start_pipeline(&PipeTensorMerger::update, this);
}

bool PipeTensorMerger::is_thread_stoped() {
  return _pipe_impl->is_pipeline_stoped();
}

void PipeTensorMerger::stop_thread() { _pipe_impl->stop_pipeline(); }

void PipeTensorMerger::update() {
  while (!is_thread_stoped()) {
    if (is_full_out()) {
      std::cout << "merger " << _pipe_impl->get_meta_info() << " output is full"
                << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
      continue;
    }

    std::vector<std::shared_ptr<SMergerInput>> input_data_list;
    int batch_size = std::min(_batch_size, _metainfo_map->size());
    while (_pipe_impl->read_multi_in(input_data_list, batch_size) <= 0) {
      if (is_thread_stoped()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
      batch_size = std::min(_batch_size, _metainfo_map->size());
    }

    // auto start_time = std::chrono::steady_clock::now();

    std::vector<at::Tensor> tensor_list;
    std::vector<cv::Mat> image_list;
    std::vector<std::string> meta_info_list;
    std::vector<SPreprocessMetaParam> meta_list;
    for (auto input_data : input_data_list) {
      tensor_list.push_back(input_data->_tensor_image);
      image_list.push_back(input_data->_image);
      meta_info_list.push_back(input_data->_meta_info);
      meta_list.push_back(input_data->_meta);
    }
    at::Tensor merged_tensor = at::cat(tensor_list, 0);
    merged_tensor = merged_tensor.to(at::Device(at::kCUDA, _device_id));

    auto output_data = get_data_out();
    output_data->_image_list = image_list;
    output_data->_meta_info_list = meta_info_list;
    output_data->_merged_tensor = merged_tensor;
    output_data->_meta_list = meta_list;

    write_out(output_data);

    // auto end_time = std::chrono::steady_clock::now();
    // auto elapse_milli = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     end_time - start_time);
    // std::cout << "merger time " << elapse_milli.count() / 1000.
    //           << " seconds." << std::endl;
  }
}