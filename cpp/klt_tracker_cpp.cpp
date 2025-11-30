#include <array>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

struct TrackObservation {
  std::array<float, 2> keypoint{};
  float depth = 0.0f;
};

struct FeatureTrack {
  int track_id = 0;
  int anchor_frame = 0;
  std::array<float, 2> anchor_keypoint{};
  float anchor_depth = 0.0f;
  std::array<float, 3> anchor_point3{};
  std::array<uint8_t, 3> anchor_color{};
  std::unordered_map<int, TrackObservation> observations;
  std::vector<int> observation_frames;
  bool active = true;

  void add_observation(int frame_idx, const cv::Point2f &pt, float depth) {
    TrackObservation obs;
    obs.keypoint = {pt.x, pt.y};
    obs.depth = depth;
    observations[frame_idx] = obs;
    if (observation_frames.empty() ||
        observation_frames.back() != frame_idx) {
      observation_frames.push_back(frame_idx);
    }
  }
};

struct KLTTrackerConfigCpp {
  int max_feature_count = 1024;
  double refill_feature_ratio = 0.8;
  double feature_suppression_radius = 8.0;

  cv::Size lk_win_size = cv::Size(15, 15);
  int lk_max_level = 5;
  int lk_max_iterations = 40;
  double lk_epsilon = 0.01;
  double lk_min_eig_threshold = 1e-3;

  double stereo_ransac_threshold = 2.0;
  double stereo_max_y_diff = 2.0;
  double min_disparity = 0.1;
  double max_depth = 40.0;

  double gftt_quality_level = 0.001;
  double gftt_min_distance = 20.0;
  int gftt_block_size = 3;
  bool gftt_use_harris_detector = false;
  double gftt_k = 0.04;

  int templ_rows = 11;
  int templ_cols = 101;
  int stripe_extra_rows = 0;
  double template_matching_tolerance = 0.15;
  bool subpixel_refinement = false;
  double stereo_min_depth = 0.15;

  int stereo_ransac_min_inliers = 8;
  double stereo_ransac_confidence = 0.999;
};

namespace {

cv::Mat to_uint8_mat(const py::array &array) {
  py::buffer_info info = array.request();
  if (info.ndim == 2) {
    return cv::Mat(static_cast<int>(info.shape[0]),
                   static_cast<int>(info.shape[1]), CV_8UC1, info.ptr);
  }
  if (info.ndim == 3 && info.shape[2] == 3) {
    return cv::Mat(static_cast<int>(info.shape[0]),
                   static_cast<int>(info.shape[1]), CV_8UC3, info.ptr);
  }
  throw std::runtime_error("Expected an HxW or HxWx3 uint8 array");
}

cv::Matx44d to_q_matrix(const py::array &q_array) {
  py::buffer_info info = q_array.request();
  if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
    throw std::runtime_error("Q must have shape (4, 4)");
  }
  const double *ptr = static_cast<const double *>(info.ptr);
  cv::Matx44d Q;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      Q(i, j) = ptr[4 * i + j];
    }
  }
  return Q;
}

std::vector<cv::Point2f> filter_keypoints(
    const std::vector<cv::Point2f> &candidates,
    const std::vector<cv::Point2f> &existing, double min_radius) {
  if (candidates.empty()) {
    return {};
  }
  if (existing.empty()) {
    return candidates;
  }

  std::vector<cv::Point2f> kept;
  std::vector<cv::Point2f> current = existing;
  const float min_radius_sq = static_cast<float>(min_radius * min_radius);

  for (const auto &pt : candidates) {
    bool too_close = false;
    for (const auto &cur : current) {
      const float dx = cur.x - pt.x;
      const float dy = cur.y - pt.y;
      if (dx * dx + dy * dy < min_radius_sq) {
        too_close = true;
        break;
      }
    }
    if (too_close) {
      continue;
    }
    kept.push_back(pt);
    current.push_back(pt);
  }
  return kept;
}

std::vector<cv::Point2f> detect_keypoints(const cv::Mat &gray,
                                          int max_features,
                                          const KLTTrackerConfigCpp &cfg) {
  std::vector<cv::Point2f> keypoints;
  cv::goodFeaturesToTrack(
      gray, keypoints, max_features, cfg.gftt_quality_level,
      cfg.gftt_min_distance, cv::Mat(), cfg.gftt_block_size,
      cfg.gftt_use_harris_detector, cfg.gftt_k);
  return keypoints;
}

struct StereoComputationResult {
  std::vector<bool> valid_mask;
  std::vector<cv::Point2f> right_points;
  std::vector<float> depths;
  std::vector<std::array<float, 3>> points3d;
};

StereoComputationResult compute_stereo_matches(
    const cv::Mat &left_gray, const cv::Mat &right_gray,
    const std::vector<cv::Point2f> &left_points, const cv::Matx44d &Q,
    const KLTTrackerConfigCpp &cfg) {
  const int n_points = static_cast<int>(left_points.size());
  StereoComputationResult result;
  result.valid_mask.assign(n_points, false);
  result.right_points.assign(n_points, cv::Point2f(0.0f, 0.0f));
  result.depths.assign(n_points, 0.0f);
  result.points3d.assign(n_points, std::array<float, 3>{0.0f, 0.0f, 0.0f});

  if (n_points == 0) {
    return result;
  }

  const int h = left_gray.rows;
  const int w = left_gray.cols;
  std::vector<bool> stereo_hit(n_points, false);

  const int templ_rows = cfg.templ_rows;
  const int templ_cols = cfg.templ_cols;
  const int stripe_rows = templ_rows + cfg.stripe_extra_rows;

  const double fx = Q(2, 3);
  const double baseline = std::abs(Q(3, 2)) > 1e-9 ? 1.0 / std::abs(Q(3, 2))
                                                   : 1.0;
  const double min_depth = std::max(cfg.stereo_min_depth, 1e-3);
  const int stripe_cols_raw =
      static_cast<int>(std::round(fx * baseline / min_depth) + templ_cols + 4);
  const int stripe_cols =
      std::max(templ_cols, std::min(stripe_cols_raw, w));

  cv::Mat right_float;
  if (cfg.subpixel_refinement) {
    right_gray.convertTo(right_float, CV_32F);
  }

  for (int i = 0; i < n_points; ++i) {
    const float x = left_points[i].x;
    const float y = left_points[i].y;
    const int rounded_x = static_cast<int>(std::round(x));
    const int rounded_y = static_cast<int>(std::round(y));

    if (rounded_x < 0 || rounded_x >= w || rounded_y < 0 || rounded_y >= h) {
      continue;
    }

    const int temp_corner_y = rounded_y - (templ_rows - 1) / 2;
    if (temp_corner_y < 0 || temp_corner_y + templ_rows > h) {
      continue;
    }

    int temp_corner_x = rounded_x - (templ_cols - 1) / 2;
    int offset_temp = 0;
    if (temp_corner_x < 0) {
      offset_temp = temp_corner_x;
      temp_corner_x = 0;
    }
    if (temp_corner_x + templ_cols > w) {
      offset_temp = (temp_corner_x + templ_cols) - w;
      temp_corner_x -= offset_temp;
      if (temp_corner_x < 0) {
        continue;
      }
    }

    const int stripe_corner_y = rounded_y - (stripe_rows - 1) / 2;
    if (stripe_corner_y < 0 || stripe_corner_y + stripe_rows > h) {
      continue;
    }

    int stripe_corner_x = rounded_x + (templ_cols - 1) / 2 - stripe_cols;
    int offset_stripe = 0;
    if (stripe_corner_x + stripe_cols > w) {
      offset_stripe = (stripe_corner_x + stripe_cols) - w;
      stripe_corner_x -= offset_stripe;
    }
    if (stripe_corner_x < 0) {
      stripe_corner_x = 0;
    }

    const cv::Rect templ_roi(temp_corner_x, temp_corner_y, templ_cols,
                             templ_rows);
    const cv::Rect stripe_roi(stripe_corner_x, stripe_corner_y, stripe_cols,
                              stripe_rows);

    if (templ_roi.x < 0 || templ_roi.y < 0 ||
        templ_roi.x + templ_roi.width > w ||
        templ_roi.y + templ_roi.height > h) {
      continue;
    }
    if (stripe_roi.x < 0 || stripe_roi.y < 0 ||
        stripe_roi.x + stripe_roi.width > w ||
        stripe_roi.y + stripe_roi.height > h) {
      continue;
    }

    const cv::Mat templ(left_gray, templ_roi);
    const cv::Mat stripe(right_gray, stripe_roi);
    if (stripe.rows < templ.rows || stripe.cols < templ.cols) {
      continue;
    }

    cv::Mat ssd;
    cv::matchTemplate(stripe, templ, ssd, cv::TM_SQDIFF);
    cv::normalize(ssd, ssd, 0, 1, cv::NORM_MINMAX);

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(ssd, &min_val, &max_val, &min_loc, &max_loc);

    const int match_x =
        min_loc.x + stripe_corner_x + (templ_cols - 1) / 2 + offset_temp;
    const int match_y = min_loc.y + stripe_corner_y + (templ_rows - 1) / 2;

    if (match_x < 0 || match_x >= w || match_y < 0 || match_y >= h) {
      continue;
    }

    cv::Point2f match_pt(static_cast<float>(match_x),
                         static_cast<float>(match_y));

    if (std::fabs(y - match_pt.y) > cfg.stereo_max_y_diff) {
      continue;
    }

    const float disparity = x - match_pt.x;
    if (disparity < cfg.min_disparity) {
      continue;
    }

    const float score = static_cast<float>(min_val);
    if (score >= cfg.template_matching_tolerance) {
      continue;
    }

    if (cfg.subpixel_refinement) {
      std::vector<cv::Point2f> pts{match_pt};
      cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                40, 0.001);
      cv::cornerSubPix(right_float, pts, cv::Size(10, 10), cv::Size(-1, -1),
                       criteria);
      match_pt = pts[0];
    }

    stereo_hit[i] = true;
    result.right_points[i] = match_pt;
  }

  std::vector<int> valid_indices;
  valid_indices.reserve(n_points);
  for (int i = 0; i < n_points; ++i) {
    if (stereo_hit[i]) {
      valid_indices.push_back(i);
    }
  }

  if (valid_indices.empty()) {
    return result;
  }

  auto reproject = [&](bool invert) {
    std::vector<double> depths(valid_indices.size(), 0.0);
    std::vector<cv::Vec3d> pts(valid_indices.size(), cv::Vec3d(0, 0, 0));
    for (size_t idx = 0; idx < valid_indices.size(); ++idx) {
      const int i = valid_indices[idx];
      const double lx = left_points[i].x;
      const double ly = left_points[i].y;
      double disparity = lx - result.right_points[i].x;
      if (invert) {
        disparity = -disparity;
      }
      const cv::Vec4d vec(lx, ly, disparity, 1.0);
      const cv::Vec4d hom = Q * vec;
      double w_val = hom[3];
      if (std::abs(w_val) < 1e-6) {
        w_val = 1e-6;
      }
      const cv::Vec3d p(hom[0] / w_val, hom[1] / w_val, hom[2] / w_val);
      pts[idx] = p;
      depths[idx] = p[2];
    }
    return std::make_pair(std::move(pts), std::move(depths));
  };

  auto reprojection = reproject(false);
  const auto &pts_valid = reprojection.first;
  const auto &depths_valid = reprojection.second;

  int positive = 0;
  int negative = 0;
  for (double d : depths_valid) {
    if (std::isfinite(d)) {
      if (d > 0.0) {
        ++positive;
      } else if (d < 0.0) {
        ++negative;
      }
    }
  }

  std::vector<cv::Vec3d> pts_adjusted = pts_valid;
  std::vector<double> depths_adjusted = depths_valid;
  if (negative > positive) {
    auto flipped = reproject(true);
    pts_adjusted = std::move(flipped.first);
    depths_adjusted = std::move(flipped.second);
  }

  result.valid_mask.assign(n_points, false);
  const size_t valid_count = valid_indices.size();
  for (size_t j = 0; j < valid_count; ++j) {
    const double depth = depths_adjusted[j];
    if (depth <= 0.0 || depth > cfg.max_depth || !std::isfinite(depth)) {
      continue;
    }
    const int idx = valid_indices[j];
    result.valid_mask[idx] = true;
    result.depths[idx] = static_cast<float>(depth);
    result.points3d[idx] = {static_cast<float>(pts_adjusted[j][0]),
                            static_cast<float>(pts_adjusted[j][1]),
                            static_cast<float>(pts_adjusted[j][2])};
  }

  return result;
}

py::dict obs_map_to_dict(const std::unordered_map<int, TrackObservation> &obs) {
  py::dict out;
  for (const auto &[id, observation] : obs) {
    out[py::int_(id)] = py::cast(observation);
  }
  return out;
}

py::dict track_map_to_dict(const std::unordered_map<int, FeatureTrack> &tracks) {
  py::dict out;
  for (const auto &[id, track] : tracks) {
    out[py::int_(id)] = py::cast(track);
  }
  return out;
}

} // namespace

class KLTFeatureTrackerCpp {
public:
  explicit KLTFeatureTrackerCpp(
      int max_feature_count = 1024, double refill_feature_ratio = 0.8,
      double feature_suppression_radius = 8.0,
      std::pair<int, int> lk_win_size = {15, 15}, int lk_max_level = 5,
      int lk_max_iterations = 40, double lk_epsilon = 0.01,
      double lk_min_eig_threshold = 1e-3, double max_depth = 40.0,
      double gftt_quality_level = 0.001, double gftt_min_distance = 20.0,
      int gftt_block_size = 3, bool gftt_use_harris_detector = false,
      double gftt_k = 0.04, int templ_rows = 11, int templ_cols = 101,
      int stripe_extra_rows = 0, double template_matching_tolerance = 0.15,
      bool subpixel_refinement = false, double stereo_min_depth = 0.15,
      double stereo_ransac_threshold = 2.0, double stereo_max_y_diff = 2.0,
      double min_disparity = 0.1,
      int stereo_ransac_min_inliers = 8,
      double stereo_ransac_confidence = 0.999)
      : config_{}, lk_criteria_(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                lk_max_iterations, lk_epsilon) {
    config_.max_feature_count = max_feature_count;
    config_.refill_feature_ratio = refill_feature_ratio;
    config_.feature_suppression_radius = feature_suppression_radius;
    config_.lk_win_size = cv::Size(lk_win_size.first, lk_win_size.second);
    config_.lk_max_level = lk_max_level;
    config_.lk_max_iterations = lk_max_iterations;
    config_.lk_epsilon = lk_epsilon;
    config_.lk_min_eig_threshold = lk_min_eig_threshold;
    config_.max_depth = max_depth;
    config_.gftt_quality_level = gftt_quality_level;
    config_.gftt_min_distance = gftt_min_distance;
    config_.gftt_block_size = gftt_block_size;
    config_.gftt_use_harris_detector = gftt_use_harris_detector;
    config_.gftt_k = gftt_k;
    config_.templ_rows = templ_rows;
    config_.templ_cols = templ_cols;
    config_.stripe_extra_rows = stripe_extra_rows;
    config_.template_matching_tolerance = template_matching_tolerance;
    config_.subpixel_refinement = subpixel_refinement;
    config_.stereo_min_depth = stereo_min_depth;
    config_.stereo_ransac_threshold = stereo_ransac_threshold;
    config_.stereo_max_y_diff = stereo_max_y_diff;
    config_.min_disparity = min_disparity;
    config_.stereo_ransac_min_inliers = stereo_ransac_min_inliers;
    config_.stereo_ransac_confidence = stereo_ransac_confidence;

    refill_threshold_ =
        static_cast<int>(std::floor(config_.refill_feature_ratio *
                                    config_.max_feature_count));
    reset();
  }

  void reset() {
    tracks_.clear();
    track_history_.clear();
    next_track_id_ = 0;
    prev_gray_.release();
    prev_points_.clear();
    prev_ids_.clear();
  }

  py::dict track_frame(const py::object &rectified_frame) {
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
        left_arr = rectified_frame.attr("left_rect").cast<py::array>();
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
        right_arr = rectified_frame.attr("right_rect").cast<py::array>();
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        q_array = rectified_frame.attr("calibration").attr("Q").cast<py::array>();

    const cv::Mat left_mat = to_uint8_mat(left_arr);
    const cv::Mat right_mat = to_uint8_mat(right_arr);
    const cv::Matx44d Q = to_q_matrix(q_array);

    cv::Mat gray_left;
    cv::Mat gray_right;
    if (left_mat.channels() == 3) {
      cv::cvtColor(left_mat, gray_left, cv::COLOR_RGB2GRAY);
    } else {
      gray_left = left_mat.clone();
    }
    if (right_mat.channels() == 3) {
      cv::cvtColor(right_mat, gray_right, cv::COLOR_RGB2GRAY);
    } else {
      gray_right = right_mat.clone();
    }

    const int frame_idx = static_cast<int>(track_history_.size());
    std::unordered_map<int, TrackObservation> frame_obs;
    std::vector<cv::Point2f> tracked_points;
    std::vector<int> tracked_ids;

    if (!prev_gray_.empty() && !prev_points_.empty() && !prev_ids_.empty()) {
      std::vector<cv::Point2f> next_points;
      std::vector<uchar> status;
      std::vector<float> err;
      cv::calcOpticalFlowPyrLK(
          prev_gray_, gray_left, prev_points_, next_points, status, err,
          config_.lk_win_size, config_.lk_max_level, lk_criteria_,
          0, config_.lk_min_eig_threshold);

      std::vector<int> candidate_ids;
      std::vector<cv::Point2f> candidate_points;
      const size_t n_prev = prev_points_.size();
      candidate_ids.reserve(n_prev);
      candidate_points.reserve(n_prev);
      for (size_t i = 0; i < n_prev; ++i) {
        if (status[i]) {
          candidate_ids.push_back(prev_ids_[i]);
          candidate_points.push_back(next_points[i]);
        }
      }

      const auto stereo = compute_stereo_matches(
          gray_left, gray_right, candidate_points, Q, config_);

      for (size_t i = 0; i < candidate_ids.size(); ++i) {
        if (!stereo.valid_mask[i]) {
          continue;
        }
        const int track_id = candidate_ids[i];
        const cv::Point2f pt = candidate_points[i];
        const float depth = stereo.depths[i];

        FeatureTrack &track = tracks_[track_id];
        track.add_observation(frame_idx, pt, depth);
        track.active = true;

        frame_obs[track_id] = track.observations[frame_idx];
        tracked_points.push_back(pt);
        tracked_ids.push_back(track_id);
      }

      std::unordered_set<int> active_set(tracked_ids.begin(), tracked_ids.end());
      for (int id : prev_ids_) {
        if (active_set.find(id) == active_set.end()) {
          tracks_[id].active = false;
        }
      }
    }

    std::vector<cv::Point2f> existing_points = tracked_points;

    auto add_new_tracks = [&](int budget) {
      if (budget <= 0) {
        return;
      }

      const int detection_quota =
          std::max(config_.max_feature_count, budget * 2);
      const auto detected =
          detect_keypoints(gray_left, detection_quota, config_);

      const auto filtered = filter_keypoints(
          detected, existing_points, config_.feature_suppression_radius);
      if (filtered.empty()) {
        return;
      }

      const auto stereo = compute_stereo_matches(
          gray_left, gray_right, filtered, Q, config_);

      std::vector<int> valid_indices;
      valid_indices.reserve(filtered.size());
      for (size_t i = 0; i < filtered.size(); ++i) {
        if (stereo.valid_mask[i]) {
          valid_indices.push_back(static_cast<int>(i));
        }
      }

      const int take =
          std::min(budget, static_cast<int>(valid_indices.size()));
      const int img_h = left_mat.rows;
      const int img_w = left_mat.cols;
      for (int i = 0; i < take; ++i) {
        const int idx = valid_indices[i];
        const cv::Point2f kp = filtered[idx];
        const float depth = stereo.depths[idx];
        const auto &pt3 = stereo.points3d[idx];

        const int ix = static_cast<int>(std::round(kp.x));
        const int iy = static_cast<int>(std::round(kp.y));
        std::array<uint8_t, 3> color{0, 0, 0};
        if (ix >= 0 && ix < img_w && iy >= 0 && iy < img_h &&
            left_mat.channels() == 3) {
          const cv::Vec3b c = left_mat.at<cv::Vec3b>(iy, ix);
          color = {c[0], c[1], c[2]};
        }

        FeatureTrack track;
        track.track_id = next_track_id_;
        track.anchor_frame = frame_idx;
        track.anchor_keypoint = {kp.x, kp.y};
        track.anchor_depth = depth;
        track.anchor_point3 = {pt3[0], pt3[1], pt3[2]};
        track.anchor_color = color;
        track.add_observation(frame_idx, kp, depth);

        tracks_[next_track_id_] = track;
        frame_obs[next_track_id_] = track.observations[frame_idx];
        tracked_points.push_back(kp);
        tracked_ids.push_back(next_track_id_);
        existing_points.push_back(kp);
        ++next_track_id_;
      }
    };

    const bool need_refill =
        static_cast<int>(tracked_ids.size()) <= refill_threshold_;
    const int budget =
        need_refill ? (config_.max_feature_count -
                       static_cast<int>(tracked_ids.size()))
                    : 0;
    add_new_tracks(budget);

    for (int id : tracked_ids) {
      tracks_[id].active = true;
    }

    prev_gray_ = gray_left;
    prev_points_ = tracked_points;
    prev_ids_ = tracked_ids;

    track_history_.push_back(frame_obs);
    return obs_map_to_dict(frame_obs);
  }

  std::pair<py::list, py::dict> track(const py::list &rectified_frames) {
    reset();
    for (const auto &item : rectified_frames) {
      track_frame(py::reinterpret_borrow<py::object>(item));
    }
    py::list history;
    for (const auto &frame_obs : track_history_) {
      history.append(obs_map_to_dict(frame_obs));
    }
    return {history, track_map_to_dict(tracks_)};
  }

  py::dict get_tracks() const { return track_map_to_dict(tracks_); }
  py::list get_track_history() const {
    py::list history;
    for (const auto &frame_obs : track_history_) {
      history.append(obs_map_to_dict(frame_obs));
    }
    return history;
  }

  py::dict config() const {
    py::dict out;
    out["max_feature_count"] = config_.max_feature_count;
    out["refill_feature_ratio"] = config_.refill_feature_ratio;
    out["feature_suppression_radius"] = config_.feature_suppression_radius;
    out["lk_win_size"] =
        py::make_tuple(config_.lk_win_size.width, config_.lk_win_size.height);
    out["lk_max_level"] = config_.lk_max_level;
    out["lk_max_iterations"] = config_.lk_max_iterations;
    out["lk_epsilon"] = config_.lk_epsilon;
    out["lk_min_eig_threshold"] = config_.lk_min_eig_threshold;
    out["stereo_ransac_threshold"] = config_.stereo_ransac_threshold;
    out["stereo_max_y_diff"] = config_.stereo_max_y_diff;
    out["min_disparity"] = config_.min_disparity;
    out["max_depth"] = config_.max_depth;
    out["gftt_quality_level"] = config_.gftt_quality_level;
    out["gftt_min_distance"] = config_.gftt_min_distance;
    out["gftt_block_size"] = config_.gftt_block_size;
    out["gftt_use_harris_detector"] = config_.gftt_use_harris_detector;
    out["gftt_k"] = config_.gftt_k;
    out["templ_rows"] = config_.templ_rows;
    out["templ_cols"] = config_.templ_cols;
    out["stripe_extra_rows"] = config_.stripe_extra_rows;
    out["template_matching_tolerance"] = config_.template_matching_tolerance;
    out["subpixel_refinement"] = config_.subpixel_refinement;
    out["stereo_min_depth"] = config_.stereo_min_depth;
    out["stereo_ransac_min_inliers"] = config_.stereo_ransac_min_inliers;
    out["stereo_ransac_confidence"] = config_.stereo_ransac_confidence;
    return out;
  }

private:
  KLTTrackerConfigCpp config_;
  cv::TermCriteria lk_criteria_;
  int refill_threshold_ = 0;

  std::unordered_map<int, FeatureTrack> tracks_;
  std::vector<std::unordered_map<int, TrackObservation>> track_history_;

  cv::Mat prev_gray_;
  std::vector<cv::Point2f> prev_points_;
  std::vector<int> prev_ids_;
  int next_track_id_ = 0;
};

PYBIND11_MODULE(klt_tracker_cpp, m) {
  m.doc() = "Fast C++ implementation of the KLT feature tracker";

  py::class_<TrackObservation>(m, "TrackObservation")
      .def(py::init<>())
      .def_readwrite("keypoint", &TrackObservation::keypoint)
      .def_readwrite("depth", &TrackObservation::depth);

  py::class_<FeatureTrack>(m, "FeatureTrack")
      .def(py::init<>())
      .def_readwrite("track_id", &FeatureTrack::track_id)
      .def_readwrite("anchor_frame", &FeatureTrack::anchor_frame)
      .def_readwrite("anchor_keypoint", &FeatureTrack::anchor_keypoint)
      .def_readwrite("anchor_depth", &FeatureTrack::anchor_depth)
      .def_readwrite("anchor_point3", &FeatureTrack::anchor_point3)
      .def_readwrite("anchor_color", &FeatureTrack::anchor_color)
      .def_readwrite("observations", &FeatureTrack::observations)
      .def_readwrite("observation_frames", &FeatureTrack::observation_frames)
      .def_readwrite("active", &FeatureTrack::active);

  py::class_<KLTFeatureTrackerCpp>(m, "KLTFeatureTrackerCpp")
      .def(py::init<int, double, double, std::pair<int, int>, int, int, double,
                    double, double, double, double, int, bool, double, int, int,
                    int, double, bool, double, double, double, double, int,
                    double>(),
           py::arg("max_feature_count") = 1024,
           py::arg("refill_feature_ratio") = 0.8,
           py::arg("feature_suppression_radius") = 8.0,
           py::arg("lk_win_size") = std::make_pair(15, 15),
           py::arg("lk_max_level") = 5, py::arg("lk_max_iterations") = 40,
           py::arg("lk_epsilon") = 0.01,
           py::arg("lk_min_eig_threshold") = 1e-3,
           py::arg("max_depth") = 40.0,
           py::arg("gftt_quality_level") = 0.001,
           py::arg("gftt_min_distance") = 20.0,
           py::arg("gftt_block_size") = 3,
           py::arg("gftt_use_harris_detector") = false,
           py::arg("gftt_k") = 0.04, py::arg("templ_rows") = 11,
           py::arg("templ_cols") = 101, py::arg("stripe_extra_rows") = 0,
           py::arg("template_matching_tolerance") = 0.15,
           py::arg("subpixel_refinement") = false,
           py::arg("stereo_min_depth") = 0.15,
           py::arg("stereo_ransac_threshold") = 2.0,
           py::arg("stereo_max_y_diff") = 2.0,
           py::arg("min_disparity") = 0.1,
           py::arg("stereo_ransac_min_inliers") = 8,
           py::arg("stereo_ransac_confidence") = 0.999)
      .def("reset", &KLTFeatureTrackerCpp::reset)
      .def("track_frame", &KLTFeatureTrackerCpp::track_frame)
      .def("track", &KLTFeatureTrackerCpp::track)
      .def_property_readonly("tracks", &KLTFeatureTrackerCpp::get_tracks)
      .def_property_readonly("track_history",
                             &KLTFeatureTrackerCpp::get_track_history)
      .def_property_readonly("config", &KLTFeatureTrackerCpp::config);
}
