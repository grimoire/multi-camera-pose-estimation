#pragma once
#include "pipeline/pipe_base.h"

template <typename T> class PipeOutput : public PipeBase<int, T> {
public:
  PipeOutput() {}
  ~PipeOutput() {}
};