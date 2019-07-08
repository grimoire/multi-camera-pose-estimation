#pragma once
#include "pipeline/pipe_process.h"
#include "pipeline/metainfo_map.h"
#include <opencv2/core/core.hpp>
#include <string>
#include <torch/script.h>

class PipeTensorSplitImpl;
class PipeTensorSplit {
public:
  PipeTensorSplit();
  ~PipeTensorSplit();

  void init_pipeline(PipeProcess *pipe_process,
                     MetaInfoMap *metainfo_map, const std::string &meta_info);

  virtual void start_thread();
  virtual void stop_thread();
  virtual bool is_thread_stoped();

protected:
  virtual void update();

private:
  PipeProcess *_pipe_process;
  MetaInfoMap *_metainfo_map;
  PipeTensorSplitImpl *_pipe_impl;
};