#pragma once
#include <boost/atomic.hpp>
#include <boost/pool/object_pool.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <iostream>
#include <memory>
#include <stack>
#include <string>

template <typename T>
inline bool stack_push(std::stack<std::shared_ptr<T>> &stack,
                       const std::shared_ptr<T> &value, int max_value) {

  if (stack.size() >= max_value) {
    return false;
  }
  stack.push(value);
  return true;
}

template <typename T>
inline bool stack_pop(std::stack<std::shared_ptr<T>> &stack,
                      std::shared_ptr<T> &value) {
  if (stack.size() == 0) {
    return false;
  }
  value = stack.top();
  stack.pop();
  return true;
}

template <typename T> void clear_stack(std::stack<std::shared_ptr<T>> &stack) {
  while (!stack.empty()) {
    stack.pop();
  }
}

template <typename IN_T, typename OUT_T> class PipeBase {

public:
  typedef IN_T in_value_type;
  typedef OUT_T out_value_type;
  typedef std::shared_ptr<in_value_type> in_value_ptr;
  typedef std::shared_ptr<out_value_type> out_value_ptr;

  // constructor
  PipeBase() {}

  // destructor
  ~PipeBase() {
    clear_in();
    clear_out();
  }

  // initial input and output pipeline
  // diable stack if size is zero
  void init_pipeline(size_t in_size, size_t out_size,
                     const std::string &meta_info = "") {
    _meta_info = meta_info;
    _input_max_size = in_size;
    _output_max_size = out_size;
    _stop_pipeline = true;
  }

  // start the thread
  template <typename UT> void start_pipeline(void (UT::*fptr)(), UT *ut) {
    if (_stop_pipeline) {
      boost::thread thrd(boost::bind(fptr, ut));
      _stop_pipeline = false;
      thrd.detach();
    }
  }

  void stop_pipeline() { _stop_pipeline = true; }

  bool is_pipeline_stoped() { return _stop_pipeline; }

  in_value_ptr getInputInstance() const { return std::make_shared<IN_T>(); }

  out_value_ptr getOutputInstance() const { return std::make_shared<OUT_T>(); }

  virtual bool is_input_full() {
    boost::lock_guard<boost::mutex> guard(_input_mutex);
    return _input_container.size() >= _input_max_size;
  }

  virtual bool is_output_full() {
    boost::lock_guard<boost::mutex> guard(_output_mutex);
    return _output_container.size() >= _output_max_size;
  }

  virtual bool is_input_empty() {
    boost::lock_guard<boost::mutex> guard(_input_mutex);
    return _input_container.size() == 0;
  }

  virtual bool is_output_empty() {
    boost::lock_guard<boost::mutex> guard(_output_mutex);
    return _output_container.size() == 0;
  }

  virtual bool write_in(const in_value_ptr &value) {
    boost::lock_guard<boost::mutex> guard(_input_mutex);
    return stack_push<IN_T>(_input_container, value, _input_max_size);
  }

  virtual bool write_out(const out_value_ptr &value) {
    boost::lock_guard<boost::mutex> guard(_output_mutex);
    return stack_push<OUT_T>(_output_container, value, _output_max_size);
  }

  virtual bool read_in(in_value_ptr &value) {
    boost::lock_guard<boost::mutex> guard(_input_mutex);
    return stack_pop<IN_T>(_input_container, value);
  }

  virtual bool read_out(out_value_ptr &value) {
    boost::lock_guard<boost::mutex> guard(_output_mutex);
    return stack_pop<OUT_T>(_output_container, value);
  }

  virtual int write_multi_in(const std::vector<in_value_ptr> &values) {
    boost::lock_guard<boost::mutex> guard(_input_mutex);
    int success_count = 0;
    for (auto value : values) {
      if (stack_push<IN_T>(_input_container, value, _input_max_size)) {
        success_count += 1;
      }
    }
    return success_count;
  }

  virtual int write_multi_out(const std::vector<out_value_ptr> &values) {
    boost::lock_guard<boost::mutex> guard(_output_mutex);
    int success_count = 0;
    for (auto value : values) {
      if (stack_push<OUT_T>(_output_container, value, _output_max_size)) {
        success_count += 1;
      }
    }
    return success_count;
  }

  virtual int read_multi_in(std::vector<in_value_ptr> &values, int read_size) {
    boost::lock_guard<boost::mutex> guard(_input_mutex);
    int success_count = 0;
    for (int i = 0; i < read_size; ++i) {
      in_value_ptr value;
      if (stack_pop<IN_T>(_input_container, value)) {
        values.push_back(value);
        success_count += 1;
      }
    }
    return success_count;
  }

  virtual int read_multi_out(std::vector<out_value_ptr> &values,
                             int read_size) {
    boost::lock_guard<boost::mutex> guard(_output_mutex);
    int success_count = 0;
    for (int i = 0; i < read_size; ++i) {
      out_value_ptr value;
      if (stack_pop<OUT_T>(_output_container, value)) {
        values.push_back(value);
        success_count += 1;
      }
    }
    return success_count;
  }

  virtual void clear_in() {
    boost::lock_guard<boost::mutex> guard(_input_mutex);
    clear_stack<IN_T>(_input_container);
  }

  virtual void clear_out() {
    boost::lock_guard<boost::mutex> guard(_output_mutex);
    clear_stack<OUT_T>(_output_container);
  }

  std::string get_meta_info() { return _meta_info; }

private:
  boost::atomic_bool _stop_pipeline;

protected:
  std::string _meta_info;
  std::stack<in_value_ptr> _input_container;
  std::stack<out_value_ptr> _output_container;
  size_t _input_max_size;
  size_t _output_max_size;
  boost::mutex _input_mutex;
  boost::mutex _output_mutex;
};