#include "multi_pose/multi_pose.h"
#include "pipeline/metainfo_map.h"
#include "pipeline/pipe_image_hkipcam.h"
#include "pipeline/pipe_image_video.h"
#include "pipeline/pipe_output.h"
#include "pipeline/pipe_postprocess.h"
#include "pipeline/pipe_preprocess.h"
#include "pipeline/pipe_process.h"
#include "pipeline/pipe_tensormerger.h"
#include "pipeline/pipe_tensorsplit.h"
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <thread>

const int NUM_SKELETON = 17;
const int EDGE[][2] = {{0, 1},   {0, 2},   {1, 3},  {2, 4},   {3, 5},
                       {4, 6},   {5, 6},   {5, 7},  {7, 9},   {6, 8},
                       {8, 10},  {5, 11},  {6, 12}, {11, 12}, {11, 13},
                       {13, 15}, {12, 14}, {14, 16}};

const cv::Scalar EDGE_COLOR[] = {
    cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),   cv::Scalar(255, 0, 0),
    cv::Scalar(0, 0, 255),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 0),   cv::Scalar(255, 0, 0),
    cv::Scalar(0, 0, 255),   cv::Scalar(0, 0, 255),   cv::Scalar(255, 0, 0),
    cv::Scalar(0, 0, 255),   cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 0),
    cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),   cv::Scalar(0, 0, 255)};

const cv::Scalar POINT_COLOR[] = {
    cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0),
    cv::Scalar(0, 0, 255),   cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0),
    cv::Scalar(0, 0, 255),   cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255)};

void result_visualize(std::shared_ptr<SPostprocessOutput> &output_data,
                      cv::Mat &vis_image, double threshold = 0.5) {
  static cv::Scalar BBOX_COLOR(255, 0, 0);
  vis_image = (output_data->_image).clone();
  at::Tensor &result_tensor = output_data->_result;
  for (int i = 0; i < result_tensor.size(0); ++i) {
    auto result = result_tensor.select(0, i);
    double score = result[4].item().to<double>();
    if (score < threshold) {
      continue;
    }
    cv::Point bbox_tl(result[0].item().to<int>(), result[1].item().to<int>());
    cv::Point bbox_br(result[2].item().to<int>(), result[3].item().to<int>());
    // cv::rectangle(vis_image, bbox_tl, bbox_br, BBOX_COLOR, 2);

    auto points = result.slice(0, 5).to(at::kInt).reshape({-1, 2});
    bool points_inbox[NUM_SKELETON] = {false};
    for (int idx = 0; idx < points.size(0); ++idx) {
      auto p = points.select(0, idx);
      int x = p[0].item().to<int>();
      int y = p[1].item().to<int>();

      if (x < bbox_tl.x || x > bbox_br.x || y < bbox_tl.y || y > bbox_br.y) {
        continue;
      }
      points_inbox[idx] = true;
      cv::circle(vis_image, cv::Point(x, y), 3, POINT_COLOR[idx], -1);
    }

    for (int idx = 0; idx < 18; ++idx) {
      auto edge = EDGE[idx];
      if (!(points_inbox[edge[0]] && points_inbox[edge[1]])) {
        continue;
      }
      auto tensor_p0 = points[edge[0]];
      auto tensor_p1 = points[edge[1]];
      cv::Point p0(tensor_p0[0].item().to<int>(),
                   tensor_p0[1].item().to<int>());
      cv::Point p1(tensor_p1[0].item().to<int>(),
                   tensor_p1[1].item().to<int>());
      cv::line(vis_image, p0, p1, EDGE_COLOR[idx], 2);
    }
  }
}

struct SHKIPCamPipe {
  std::shared_ptr<PipeImageHKIPCam> ptr_pipe_image;
  std::shared_ptr<PipePreprocess> ptr_pipe_preprocess;
  std::shared_ptr<PipePostprocess> ptr_pipe_postprocess;
};

SHKIPCamPipe
init_hkipc_pipeline(const std::string &ip, const std::string &username,
                    const std::string &password, long port, long channel,
                    long streamtype, int device_id,
                    PipeTensorMerger *ptr_pipe_tensormerger,
                    MultiPose *ptr_mp_model, MetaInfoMap *ptr_metainfo_map,
                    PipeOutput<SPostprocessOutput> *ptr_output_pipeline,
                    const std::string &meta_info) {
  SHKIPCamPipe video_pipe;
  video_pipe.ptr_pipe_image = std::make_shared<PipeImageHKIPCam>();
  video_pipe.ptr_pipe_preprocess = std::make_shared<PipePreprocess>();
  video_pipe.ptr_pipe_postprocess = std::make_shared<PipePostprocess>();

  video_pipe.ptr_pipe_image->init_pipeline(10, meta_info);
  video_pipe.ptr_pipe_image->init_camera(ip, username, password, port, channel,
                                         streamtype);
  video_pipe.ptr_pipe_image->set_cuda_device_id(device_id);
  video_pipe.ptr_pipe_image->set_frame_rate(21);

  video_pipe.ptr_pipe_preprocess->initPipeline(video_pipe.ptr_pipe_image.get(),
                                               ptr_pipe_tensormerger,
                                               ptr_mp_model, meta_info);
  video_pipe.ptr_pipe_postprocess->init_pipeline(
      ptr_mp_model, 5, ptr_output_pipeline, meta_info);

  video_pipe.ptr_pipe_image->start_thread();
  video_pipe.ptr_pipe_preprocess->start_thread();
  video_pipe.ptr_pipe_postprocess->start_thread();

  std::map<std::string, void *> pipe_map;
  pipe_map["image"] = static_cast<void *>(video_pipe.ptr_pipe_image.get());
  pipe_map["preprocess"] =
      static_cast<void *>(video_pipe.ptr_pipe_preprocess.get());
  pipe_map["postprocess"] =
      static_cast<void *>(video_pipe.ptr_pipe_postprocess.get());
  ptr_metainfo_map->set(meta_info, pipe_map);
  return video_pipe;
}

void pipeline_test() {

  std::string ip1 = "192.168.1.65";
  std::string ip2 = "192.168.1.61";
  std::vector<std::string> ip_list;
  ip_list.push_back(ip1);
  ip_list.push_back(ip2);
  std::string username = "admin";
  std::string password = "q1w2e3r4";
  long port = 8000;
  long channel = 1;
  long streamtype = 1;

  int num_device = 2;
  int batch_size = 8;
  int num_pipelines = 4;
  bool video_test = false;

  std::string model_path0 = "/home/deeplning/Workspace/multi-camera-pose-estimation/models/multi_pose_gpu0.pt";
  std::string model_path1 = "/home/deeplning/Workspace/multi-camera-pose-estimation/models/multi_pose_gpu1.pt";
  std::vector<std::string> model_path_list;
  model_path_list.push_back(model_path0);
  model_path_list.push_back(model_path1);

  std::string video_path =
      "~/Workspace/CenterNet-Demo/cam131.avi";

  std::vector<std::shared_ptr<MultiPose>> mp_model_list;
  std::vector<std::shared_ptr<PipeTensorMerger>> pipe_merger_list;
  std::vector<std::shared_ptr<PipeProcess>> pipe_process_list;
  std::vector<std::shared_ptr<PipeTensorSplit>> pipe_split_list;
  std::vector<std::shared_ptr<MetaInfoMap>> metainfo_map_list;

  // init device and global pipeline
  for (int device_id = 0; device_id < num_device; ++device_id) {
    std::shared_ptr<MultiPose> ptr_mp_model = std::make_shared<MultiPose>();
    ptr_mp_model->get_param()._device_id = device_id;
    auto model_path = model_path_list[device_id];
    bool success = ptr_mp_model->load_model(model_path);
    if (success) {
      std::cout << "load model success" << std::endl;
    } else {
      std::cout << "load model failed" << std::endl;
      return;
    }

    std::shared_ptr<PipeTensorMerger> ptr_pipe_tensormerger =
        std::make_shared<PipeTensorMerger>();
    std::shared_ptr<PipeProcess> ptr_pipe_process =
        std::make_shared<PipeProcess>();
    std::shared_ptr<PipeTensorSplit> ptr_pipe_tensorsplit =
        std::make_shared<PipeTensorSplit>();
    std::shared_ptr<MetaInfoMap> ptr_metainfo_map =
        std::make_shared<MetaInfoMap>();

    ptr_pipe_tensormerger->init_pipeline(ptr_metainfo_map.get(), device_id,
                                         batch_size, batch_size * 5, 10,
                                         "merger_" + std::to_string(device_id));
    ptr_pipe_process->init_pipeline(ptr_mp_model.get(),
                                    ptr_pipe_tensormerger.get(), 5,
                                    "process_" + std::to_string(device_id));
    ptr_pipe_tensorsplit->init_pipeline(ptr_pipe_process.get(),
                                        ptr_metainfo_map.get(),
                                        "split_" + std::to_string(device_id));

    ptr_pipe_tensormerger->start_thread();
    ptr_pipe_process->start_thread();
    ptr_pipe_tensorsplit->start_thread();

    mp_model_list.push_back(ptr_mp_model);
    pipe_merger_list.push_back(ptr_pipe_tensormerger);
    pipe_process_list.push_back(ptr_pipe_process);
    pipe_split_list.push_back(ptr_pipe_tensorsplit);
    metainfo_map_list.push_back(ptr_metainfo_map);
  }

  // init output
  PipeOutput<SPostprocessOutput> pipe_output;
  pipe_output.init_pipeline(0, 2 * batch_size, "output");

  std::vector<SHKIPCamPipe> hkipc_pipe_list;
  std::map<std::string, int> meta_to_id;
  for (int i = 0; i < num_pipelines; ++i) {
    int device_id = i % num_device;
    MultiPose *ptr_mp_model = mp_model_list[device_id].get();
    PipeTensorMerger *ptr_pipe_tensormerger = pipe_merger_list[device_id].get();
    PipeProcess *ptr_pipe_process = pipe_process_list[device_id].get();
    PipeTensorSplit *ptr_pipe_tensorsplit = pipe_split_list[device_id].get();
    MetaInfoMap *ptr_metainfo_map = metainfo_map_list[device_id].get();
    std::string meta_info = "thread_" + std::to_string(i);
    meta_to_id[meta_info] = i;

    int ip_id = i%ip_list.size();
    std::string ip = ip_list[ip_id];
    std::cout << "creating pipeline " << meta_info << std::endl;
    if (video_test) {
      // unavailable
    } else {
      SHKIPCamPipe video_pipe =
          init_hkipc_pipeline(ip, username, password, port, channel, streamtype,
                              device_id, ptr_pipe_tensormerger, ptr_mp_model,
                              ptr_metainfo_map, &pipe_output, meta_info);
      hkipc_pipe_list.push_back(video_pipe);
    }
  }

  // compute vis_img size
  int wind_w = 640;
  int wind_h = 480;
  int wind_rows = 1;
  if (num_pipelines > 2) {
    wind_rows = int(round(sqrt(num_pipelines)));
  }
  int wind_cols = int(ceil(num_pipelines / double(wind_rows)));
  int full_w = wind_w * wind_cols;
  int full_h = wind_h * wind_rows;
  double wind_ratio = std::min(1920. / full_w, std::min(1080. / full_h, 1.));
  wind_w = int(wind_w * wind_ratio);
  wind_h = int(wind_h * wind_ratio);
  cv::Mat vis_img =
      cv::Mat::zeros(wind_h * wind_rows, wind_w * wind_cols, CV_8UC3);

  while (true) {
    // auto start_time = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<SPostprocessOutput>> output_list;
    while (pipe_output.read_multi_out(output_list, num_pipelines) <= 0) {
    }
    pipe_output.clear_out();
    for (int i = 0; i < output_list.size(); ++i) {
      std::shared_ptr<SPostprocessOutput> output;
      output = output_list[i];
      cv::Mat tmp_img;
      result_visualize(output, tmp_img);
      cv::resize(tmp_img, tmp_img, cv::Size(wind_w, wind_h));
      auto meta_info = output->_meta_info;
      int meta_id = meta_to_id[meta_info];

      cv::putText(tmp_img, meta_info, cv::Point(20, 20),
                  cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
      int col_id = int(meta_id % wind_cols);
      int row_id = int(meta_id / wind_cols);
      cv::Rect tmp_rect(col_id * wind_w, row_id * wind_h, wind_w, wind_h);
      tmp_img.copyTo(vis_img(tmp_rect));
    }
    cv::imshow("image", vis_img);
    cv::waitKey(1);
  }
}

int main() {
  pipeline_test();

  return 0;
}