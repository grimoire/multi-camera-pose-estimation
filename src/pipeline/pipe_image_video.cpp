#include "pipeline/pipe_image_video.h"
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

void PipeImageVideo::update() {
  cv::VideoCapture video_capture;
  video_capture.open(_video_path);
  std::cout << get_meta_info() << std::endl;
  if (!video_capture.isOpened()) {
    std::cout << "video open failed." << std::endl;
    return;
  }
  int max_frame_time = int(1000 / get_frame_rate());
  while (!is_thread_stoped()) {
    if (is_full_out()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(max_frame_time));
      continue;
    }
    auto start_time = std::chrono::steady_clock::now();

    auto data = get_data_inst();
    bool read_success = video_capture.read(data->_image);
    data->_meta_info = get_meta_info();
    if (!read_success) {
      video_capture.release();
      break;
    }
    auto end_time = std::chrono::steady_clock::now();
    auto elapse_milli = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    if (elapse_milli.count() < max_frame_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(max_frame_time) -
                                  elapse_milli);
    } else {
      // std::cout << elapse_milli.count() << std::endl;
    }
    write_out(data);
  }
  video_capture.release();
}