#include "multi_pose/multi_pose.h"
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <torch/script.h>

// {'hp_offset': 2, 'hps': 34, 'hm': 1, 'hm_hp': 17, 'wh': 2, 'reg': 2}

static cv::Scalar kImageMean(0.408, 0.447, 0.470);
static cv::Scalar kImageStd(0.289, 0.274, 0.278);

inline at::Tensor _nms(const at::Tensor &heat, int kernel = 3) {
  int pad = floor((kernel - 1) / 2);
  auto hmax = at::max_pool2d(heat, {kernel, kernel}, {1, 1}, {pad, pad});
  auto keep = (hmax == heat).to(at::kFloat);
  return heat * keep;
}

inline at::Tensor _gather_feat(const at::Tensor &feat, const at::Tensor &ind) {
  int dim = feat.size(2);
  auto expand_ind = ind.unsqueeze(2).expand({ind.size(0), ind.size(1), dim});
  return feat.gather(1, expand_ind);
}

inline at::Tensor _tranpose_and_gather_feat(const at::Tensor &feat,
                                            const at::Tensor &ind) {
  auto ret_feat = feat.permute({0, 2, 3, 1}).contiguous();
  ret_feat = ret_feat.view({ret_feat.size(0), -1, ret_feat.size(3)});
  ret_feat = _gather_feat(ret_feat, ind);
  return ret_feat;
}
inline void _topk_channel(const at::Tensor &scores, int K,
                          at::Tensor &topk_scores, at::Tensor &topk_inds,
                          at::Tensor &topk_ys, at::Tensor &topk_xs) {
  int batch = scores.size(0);
  int cat = scores.size(1);
  int height = scores.size(2);
  int width = scores.size(3);

  auto topk_c_tuple = at::topk(scores.view({batch, cat, -1}), K);
  topk_scores = std::get<0>(topk_c_tuple);
  topk_inds = std::get<1>(topk_c_tuple);

  topk_inds = topk_inds % (height * width);
  topk_ys = (topk_inds / width).to(at::kFloat);
  topk_xs = (topk_inds % width).to(at::kFloat);
}

inline void _topk(const at::Tensor &scores, int K, at::Tensor &topk_score,
                  at::Tensor &topk_inds, at::Tensor &topk_clses,
                  at::Tensor &topk_ys, at::Tensor &topk_xs) {
  int batch = scores.size(0);
  int cat = scores.size(1);
  int height = scores.size(2);
  int width = scores.size(3);

  auto topk_c_tuple = at::topk(scores.view({batch, cat, -1}), K);
  auto topk_scores = std::get<0>(topk_c_tuple);
  topk_inds = std::get<1>(topk_c_tuple);

  topk_inds = topk_inds % (height * width);
  topk_ys = (topk_inds / width).to(at::kFloat);
  topk_xs = (topk_inds % width).to(at::kFloat);

  auto topk_tuple = at::topk(topk_scores.view({batch, -1}), K);
  topk_score = std::get<0>(topk_tuple);
  auto topk_ind = std::get<1>(topk_tuple);

  topk_clses = (topk_ind / K).to(at::kInt);
  topk_inds =
      _gather_feat(topk_inds.view({batch, -1, 1}), topk_ind).view({batch, K});
  topk_ys =
      _gather_feat(topk_ys.view({batch, -1, 1}), topk_ind).view({batch, K});
  topk_xs =
      _gather_feat(topk_xs.view({batch, -1, 1}), topk_ind).view({batch, K});
}

typedef c10::intrusive_ptr<c10::ivalue::Tuple> forward_result_t;
class MultiPoseImpl {
private:
  torch::jit::script::Module _module;
  SMultiPoseParam _param;

public:
  bool load_model(const std::string &path);

  void set_param(SMultiPoseParam &param) { _param = param; }

  SMultiPoseParam &get_param() { return _param; }

  at::Tensor preprocess(const cv::Mat &image, SPreprocessMetaParam &meta);

  forward_result_t module_forward(const at::Tensor &image_tensor);

  at::Tensor multi_pose_decode(const forward_result_t &forward_result,
                               int K = 100);

  at::Tensor postprocess(const at::Tensor &forward_result,
                         const SPreprocessMetaParam &meta);
};

bool MultiPoseImpl::load_model(const std::string &path) {
  try {
    _module = torch::jit::load(path);
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

at::Tensor MultiPoseImpl::preprocess(const cv::Mat &image,
                                     SPreprocessMetaParam &meta) {
  double resize_ratio =
      std::min(double(_param._process_size) / double(image.rows),
               double(_param._process_size) / double(image.cols));
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(), resize_ratio, resize_ratio);
  resized_image.convertTo(resized_image, CV_32FC3, 1. / 255, 0);
  resized_image = (resized_image - kImageMean);
  cv::divide(resized_image, kImageStd, resized_image);
  int pad_top = (_param._process_size - resized_image.rows) >> 1;
  int pad_bottom = (_param._process_size - resized_image.rows) - pad_top;
  int pad_left = (_param._process_size - resized_image.cols) >> 1;
  int pad_right = (_param._process_size - resized_image.cols) - pad_left;
  cv::copyMakeBorder(resized_image, resized_image, pad_top, pad_bottom,
                     pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
  at::Tensor tensor_image = torch::from_blob(
      resized_image.data, {1, resized_image.rows, resized_image.cols, 3},
      at::kFloat);
  tensor_image = tensor_image.permute({0, 3, 1, 2});

  meta._center[0] = image.cols / 2;
  meta._center[1] = image.rows / 2;
  meta._scale = std::max(image.cols, image.rows);
  meta._out_height = _param._process_size / 4;
  meta._out_width = _param._process_size / 4;

  return tensor_image.clone();
}

forward_result_t MultiPoseImpl::module_forward(const at::Tensor &image_tensor) {
  auto input_tensor =
      image_tensor.to(at::Device(torch::kCUDA, _param._device_id));
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor);
  return _module.forward(inputs).toTuple();
}

at::Tensor
MultiPoseImpl::multi_pose_decode(const forward_result_t &forward_result,
                                 int K) {
  // {'hp_offset': 2, 'hps': 34, 'hm': 1, 'hm_hp': 17, 'wh': 2, 'reg': 2}
  //['hm', 'hm_hp', 'hp_offset', 'hps', 'reg', 'wh']

  auto hm = (forward_result->elements())[0].toTensor();
  auto hm_hp = (forward_result->elements())[1].toTensor();
  auto hp_offset = (forward_result->elements())[2].toTensor();
  auto kps = (forward_result->elements())[3].toTensor();
  auto reg = (forward_result->elements())[4].toTensor();
  auto wh = (forward_result->elements())[5].toTensor();

  auto heat = hm.sigmoid_();
  hm_hp = hm_hp.sigmoid_();

  int batch = heat.size(0);
  int cat = heat.size(1);
  int height = heat.size(2);
  int width = heat.size(3);
  int num_joints = floor(kps.size(1) / 2);

  heat = _nms(heat);

  at::Tensor scores, inds, clses, ys, xs;
  _topk(heat, K, scores, inds, clses, ys, xs);

  auto gather_kps = _tranpose_and_gather_feat(kps, inds);
  gather_kps = gather_kps.view({batch, K, num_joints * 2});
  gather_kps.slice(-1, 0, gather_kps.size(-1), 2) +=
      xs.view({batch, K, 1}).expand({batch, K, num_joints});
  gather_kps.slice(-1, 1, gather_kps.size(-1), 2) +=
      ys.view({batch, K, 1}).expand({batch, K, num_joints});

  auto gather_reg = _tranpose_and_gather_feat(reg, inds);
  gather_reg = gather_reg.view({batch, K, 2});
  xs = xs.view({batch, K, 1}) + gather_reg.slice(2, 0, 1);
  ys = ys.view({batch, K, 1}) + gather_reg.slice(2, 1, 2);

  auto gather_wh = _tranpose_and_gather_feat(wh, inds);
  gather_wh = gather_wh.view({batch, K, 2});
  clses = clses.view({batch, K, 1}).to(at::kFloat);
  scores = scores.view({batch, K, 1});

  auto bboxes = at::cat(
      {xs - gather_wh.slice(-1, 0, 1) / 2, ys - gather_wh.slice(-1, 1, 2) / 2,
       xs + gather_wh.slice(-1, 0, 1) / 2, ys + gather_wh.slice(-1, 1, 2) / 2},
      2);

  auto nms_hm_hp = _nms(hm_hp);
  double thresh = 0.1;
  gather_kps = gather_kps.view({batch, K, num_joints, 2})
                   .permute({0, 2, 1, 3})
                   .contiguous();
  auto reg_kps = gather_kps.unsqueeze(3).expand({batch, num_joints, K, K, 2});
  at::Tensor hm_score, hm_inds, hm_ys, hm_xs;
  _topk_channel(nms_hm_hp, K, hm_score, hm_inds, hm_ys, hm_xs);

  auto gather_hp_offset =
      _tranpose_and_gather_feat(hp_offset, hm_inds.view({batch, -1}));
  gather_hp_offset = gather_hp_offset.view({batch, num_joints, K, 2});
  hm_xs = hm_xs + gather_hp_offset.select(3, 0);
  hm_ys = hm_ys + gather_hp_offset.select(3, 1);

  auto mask = (hm_score > thresh).to(at::kFloat);
  hm_score = (1 - mask) * -1 + mask * hm_score;
  hm_ys = (1 - mask) * (-10000) + mask * hm_ys;
  hm_xs = (1 - mask) * (-10000) + mask * hm_xs;
  auto hm_kps = at::stack({hm_xs, hm_ys}, -1)
                    .unsqueeze(2)
                    .expand({batch, num_joints, K, K, 2});

  auto dist = ((reg_kps - hm_kps) * (reg_kps - hm_kps)).sum(4).sqrt();
  auto dist_min_result = dist.min(3);
  auto min_dist = std::get<0>(dist_min_result);
  auto min_ind = std::get<1>(dist_min_result);
  hm_score = hm_score.gather(2, min_ind).unsqueeze(-1);
  min_dist = min_dist.unsqueeze(-1);
  min_ind = min_ind.view({batch, num_joints, K, 1, 1})
                .expand({batch, num_joints, K, 1, 2});
  hm_kps = hm_kps.gather(3, min_ind);
  hm_kps = hm_kps.view({batch, num_joints, K, 2});

  auto l = bboxes.select(2, 0)
               .view({batch, 1, K, 1})
               .expand({batch, num_joints, K, 1});
  auto t = bboxes.select(2, 1)
               .view({batch, 1, K, 1})
               .expand({batch, num_joints, K, 1});
  auto r = bboxes.select(2, 2)
               .view({batch, 1, K, 1})
               .expand({batch, num_joints, K, 1});
  auto b = bboxes.select(2, 3)
               .view({batch, 1, K, 1})
               .expand({batch, num_joints, K, 1});

  mask = (hm_kps.slice(-1, 0, 1) < l) + (hm_kps.slice(-1, 0, 1) > r) +
         (hm_kps.slice(-1, 1, 2) < t) + (hm_kps.slice(-1, 1, 2) > b) +
         (hm_score < thresh) + (min_dist > (at::max(b - t, r - l) * 0.3));
  mask = (mask > 0).to(at::kFloat).expand({batch, num_joints, K, 2});

  gather_kps = (1 - mask) * hm_kps + mask * gather_kps;
  gather_kps = gather_kps.permute({0, 2, 1, 3})
                   .contiguous()
                   .view({batch, K, num_joints * 2});
  auto detections = at::cat({bboxes, scores, gather_kps, clses}, 2);
  return detections;
}

at::Tensor MultiPoseImpl::postprocess(const at::Tensor &forward_result,
                                      const SPreprocessMetaParam &meta) {
  // auto result_device = forward_result.device();
  auto dets = forward_result.view({1, -1, forward_result.size(2)});
  float xoffset = meta._center[0] - meta._scale / 2;
  float yoffset = meta._center[1] - meta._scale / 2;
  auto det_offset =
      at::tensor({xoffset, yoffset}, forward_result.options()).view({1, 1, -1});
  float xscale = meta._scale / meta._out_width;
  float yscale = meta._scale / meta._out_height;
  auto det_scale =
      at::tensor({xscale, yscale}, forward_result.options()).view({1, 1, -1});
  auto bbox = dets.slice(2, 0, 4).clone();
  auto bbox_view = bbox.view({bbox.size(0), -1, 2});
  bbox_view *= det_scale;
  bbox_view += det_offset;
  auto pts = dets.slice(2, 5, 39).clone();
  auto pts_view = pts.view({pts.size(0), -1, 2});
  pts_view *= det_scale;
  pts_view += det_offset;
  auto top_preds = at::cat({bbox, dets.slice(2, 4, 5), pts}, 2);
  return top_preds;
}

MultiPose::MultiPose() : _impl(nullptr) { _impl = new MultiPoseImpl(); }

MultiPose::~MultiPose() {
  if (_impl) {
    delete _impl;
    _impl = nullptr;
  }
}

bool MultiPose::load_model(const std::string &model_path) {
  assert(_impl != nullptr);
  return _impl->load_model(model_path);
}

void MultiPose::set_param(SMultiPoseParam &param) { _impl->set_param(param); }

SMultiPoseParam &MultiPose::get_param() { return _impl->get_param(); }

at::Tensor MultiPose::preprocess(const cv::Mat &image,
                                 SPreprocessMetaParam &meta) {
  return _impl->preprocess(image, meta);
}

at::Tensor MultiPose::process(const at::Tensor &tensor, int K) {
  auto forward_result = _impl->module_forward(tensor);

  return _impl->multi_pose_decode(forward_result, K);
}

at::Tensor MultiPose::postprocess(const at::Tensor &forward_result,
                                  const SPreprocessMetaParam &meta) {
  return _impl->postprocess(forward_result, meta);
}