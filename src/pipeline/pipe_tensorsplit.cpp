#include "pipeline/pipe_tensorsplit.h"
#include "pipeline/pipe_base.h"
#include "pipeline/pipe_postprocess.h"
#include <chrono>
#include <thread>

class PipeTensorSplitImpl : public PipeBase<int, int> {
public:
  PipeTensorSplitImpl() {}
  ~PipeTensorSplitImpl() {}
};

PipeTensorSplit::PipeTensorSplit()
    : _pipe_impl(nullptr), _pipe_process(nullptr), _metainfo_map(nullptr) {
  _pipe_impl = new PipeTensorSplitImpl();
}

PipeTensorSplit::~PipeTensorSplit() {
  if (_pipe_impl != nullptr) {
    delete _pipe_impl;
  }
}

void PipeTensorSplit::init_pipeline(PipeProcess *pipe_process,
                                    MetaInfoMap *metainfo_map,
                                    const std::string &meta_info) {
  _pipe_process = pipe_process;
  _metainfo_map = metainfo_map;
  _pipe_impl->init_pipeline(0, 0, meta_info);
}

void PipeTensorSplit::start_thread() {
  _pipe_impl->start_pipeline(&PipeTensorSplit::update, this);
}

bool PipeTensorSplit::is_thread_stoped() {
  return _pipe_impl->is_pipeline_stoped();
}

void PipeTensorSplit::stop_thread() { _pipe_impl->stop_pipeline(); }

void PipeTensorSplit::update() {
  if(_pipe_process==nullptr){
    std::cout<< "please init precess first"<<std::endl;
  }
  while (!is_thread_stoped()) {
    auto input_data = _pipe_process->get_data_out();
    while (!_pipe_process->read_out(input_data)) {
      if (is_thread_stoped()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 25));
    }

    // auto start_time = std::chrono::steady_clock::now();

    _pipe_process->clear_out();

    auto batch_result = (input_data->_result).to(at::kCPU);
    int batch_size = input_data->_meta_info_list.size();
    // write_out(output_data);
    std::vector<void *> params;
    params.push_back(static_cast<void *>(&input_data));
    params.push_back(static_cast<void *>(&batch_result));
    params.push_back(nullptr);
    for (int i = 0; i < batch_size; ++i) {
      params[2] = static_cast<void *>(&i);
      std::string meta_info = input_data->_meta_info_list[i];
      _metainfo_map->process(
          meta_info,
          [&](std::map<std::string, void *> &meta_map,
              const std::vector<void *> &meta_params) {
            void *pipe_ptr = meta_map["postprocess"];
            if (pipe_ptr == nullptr) {
              return;
            }
            auto pipe_postprocess = static_cast<PipePostprocess *>(pipe_ptr);
            if (pipe_postprocess->is_full_in()) {
              return;
            }
            auto output_data = pipe_postprocess->get_data_in();
            auto meta_input_data = *(
                static_cast<std::shared_ptr<SProcessOutput> *>(meta_params[0]));
            auto meta_batch_result = *(
                static_cast<at::Tensor *>(meta_params[1]));
            auto idx = *(static_cast<int *>(meta_params[2]));
            output_data->_image = meta_input_data->_image_list[idx];
            output_data->_meta_info = meta_input_data->_meta_info_list[idx];
            output_data->_meta = meta_input_data->_meta_list[idx];
            output_data->_result = meta_batch_result.slice(0,
            idx, idx + 1);
            pipe_postprocess->write_in(output_data);
          },
          params);
    }

    // auto end_time = std::chrono::steady_clock::now();
    // auto elapse_milli = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     end_time - start_time);
    // std::cout << "split time " << elapse_milli.count() / 1000.
    //           << " seconds." << std::endl;
  }
}
