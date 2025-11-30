#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Compute sparse stereo matches for rectified images using
 * template matching along epipolar lines.
 *
 * Args:
 *   left_img  : HxW uint8 numpy array (grayscale, rectified left image)
 *   right_img : HxW uint8 numpy array (grayscale, rectified right image)
 *   left_points: Nx2 float32 numpy array of [x, y] keypoints in left image
 *   templ_rows, templ_cols: template size around each left point
 *   stripe_rows, stripe_cols: search stripe size in right image
 *   stereo_max_y_diff: max |y_left - y_right| for epipolar constraint
 *   min_disparity: min allowed disparity (x_left - x_right) in pixels
 *   template_matching_tolerance: max allowed normalized SSD score
 *   subpixel_refinement: if true, refine match with cornerSubPix
 *
 * Returns (valid_mask, right_points, scores):
 *   valid_mask  : N bool array (True if match found and passed all checks)
 *   right_points: Nx2 float32 array of right image keypoints (x, y)
 *   scores      : N float32 array of SSD scores (lower is better, -1 for
 * invalid)
 */
py::tuple compute_sparse_stereo(py::array_t<uint8_t> left_img,
                                py::array_t<uint8_t> right_img,
                                py::array_t<float> left_points, int templ_rows,
                                int templ_cols, int stripe_rows,
                                int stripe_cols, float stereo_max_y_diff,
                                float min_disparity,
                                float template_matching_tolerance,
                                bool subpixel_refinement) {
  // --- Validate & wrap images as cv::Mat ---
  py::buffer_info left_info = left_img.request();
  py::buffer_info right_info = right_img.request();

  if (left_info.ndim != 2 || right_info.ndim != 2) {
    throw std::runtime_error(
        "left_img and right_img must be 2D grayscale arrays");
  }
  if (left_info.shape[0] != right_info.shape[0] ||
      left_info.shape[1] != right_info.shape[1]) {
    throw std::runtime_error("left_img and right_img must have same size");
  }

  int h = static_cast<int>(left_info.shape[0]);
  int w = static_cast<int>(left_info.shape[1]);

  // Assume C-contiguous HxW
  auto *left_ptr = static_cast<uint8_t *>(left_info.ptr);
  auto *right_ptr = static_cast<uint8_t *>(right_info.ptr);

  cv::Mat left_mat(h, w, CV_8UC1, left_ptr);
  cv::Mat right_mat(h, w, CV_8UC1, right_ptr);

  // --- Validate & wrap points ---
  py::buffer_info pts_info = left_points.request();
  if (pts_info.ndim != 2 || pts_info.shape[1] != 2) {
    throw std::runtime_error("left_points must have shape (N, 2)");
  }
  const int N = static_cast<int>(pts_info.shape[0]);
  auto *pts_ptr = static_cast<float *>(pts_info.ptr);

  // --- Outputs ---
  py::array_t<bool> valid_mask({N});
  py::array_t<float> right_pts({N, 2});
  py::array_t<float> scores({N});

  auto valid_buf = valid_mask.request();
  auto right_buf = right_pts.request();
  auto scores_buf = scores.request();

  auto *valid_ptr = static_cast<bool *>(valid_buf.ptr);
  auto *right_out = static_cast<float *>(right_buf.ptr); // N*2
  auto *scores_out = static_cast<float *>(scores_buf.ptr);

  // Initialize outputs
  for (int i = 0; i < N; ++i) {
    valid_ptr[i] = false;
    scores_out[i] = -1.0f;
    right_out[2 * i + 0] = 0.0f;
    right_out[2 * i + 1] = 0.0f;
  }

  // Clamp stripe_cols to image width
  if (stripe_cols > w)
    stripe_cols = w;
  if (stripe_cols < templ_cols)
    stripe_cols = templ_cols;

  // --- Main loop over keypoints ---
  for (int i = 0; i < N; ++i) {
    float x = pts_ptr[2 * i + 0];
    float y = pts_ptr[2 * i + 1];

    int rounded_x = static_cast<int>(std::round(x));
    int rounded_y = static_cast<int>(std::round(y));

    // Skip points outside image
    if (rounded_x < 0 || rounded_x >= w || rounded_y < 0 || rounded_y >= h) {
      continue;
    }

    // --- Place template in left image ---
    int temp_corner_y = rounded_y - (templ_rows - 1) / 2;
    if (temp_corner_y < 0 || temp_corner_y + templ_rows > h) {
      continue; // template would go out vertically
    }

    int temp_corner_x = rounded_x - (templ_cols - 1) / 2;
    int offset_temp = 0;

    if (temp_corner_x < 0) {
      offset_temp = temp_corner_x; // negative
      temp_corner_x = 0;
    }
    if (temp_corner_x + templ_cols > w) {
      offset_temp = (temp_corner_x + templ_cols) - w;
      temp_corner_x -= offset_temp;
      if (temp_corner_x < 0) {
        // cannot place template fully inside image
        continue;
      }
    }

    cv::Rect templ_roi(temp_corner_x, temp_corner_y, templ_cols, templ_rows);
    if (templ_roi.x < 0 || templ_roi.y < 0 ||
        templ_roi.x + templ_roi.width > w ||
        templ_roi.y + templ_roi.height > h) {
      continue;
    }
    cv::Mat templ(left_mat, templ_roi);

    // --- Place stripe in right image ---
    int stripe_corner_y = rounded_y - (stripe_rows - 1) / 2;
    if (stripe_corner_y < 0 || stripe_corner_y + stripe_rows > h) {
      continue; // stripe would go out vertically
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

    cv::Rect stripe_roi(stripe_corner_x, stripe_corner_y, stripe_cols,
                        stripe_rows);
    if (stripe_roi.x < 0 || stripe_roi.y < 0 ||
        stripe_roi.x + stripe_roi.width > w ||
        stripe_roi.y + stripe_roi.height > h) {
      continue;
    }
    cv::Mat stripe(right_mat, stripe_roi);

    if (stripe.rows < templ.rows || stripe.cols < templ.cols) {
      continue; // stripe too small for template matching
    }

    // --- Template matching (SSD) ---
    cv::Mat result;
    cv::matchTemplate(stripe, templ, result, cv::TM_SQDIFF);
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    // Map match location back to image coordinates
    int match_x =
        min_loc.x + stripe_corner_x + (templ_cols - 1) / 2 + offset_temp;
    int match_y = min_loc.y + stripe_corner_y + (templ_rows - 1) / 2;

    if (match_x < 0 || match_x >= w || match_y < 0 || match_y >= h) {
      continue;
    }

    cv::Point2f match_pt(static_cast<float>(match_x),
                         static_cast<float>(match_y));

    // Epipolar y constraint
    if (std::fabs(y - match_pt.y) > stereo_max_y_diff) {
      continue;
    }

    float disparity = x - match_pt.x;
    if (disparity < min_disparity) {
      continue;
    }

    // Score threshold (lower is better)
    float score = static_cast<float>(min_val);
    if (score >= template_matching_tolerance) {
      continue;
    }

    // Subpixel refinement
    if (subpixel_refinement) {
      cv::Mat right_float;
      right_mat.convertTo(right_float, CV_32F);

      std::vector<cv::Point2f> pts;
      pts.push_back(match_pt);

      cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                40, 0.001);
      cv::cornerSubPix(right_float, pts, cv::Size(10, 10), cv::Size(-1, -1),
                       criteria);
      match_pt = pts[0];
    }

    // Write outputs
    valid_ptr[i] = true;
    right_out[2 * i + 0] = match_pt.x;
    right_out[2 * i + 1] = match_pt.y;
    scores_out[i] = score;
  }

  return py::make_tuple(valid_mask, right_pts, scores);
}

PYBIND11_MODULE(stereo_matching, m) {
  m.doc() = "Fast stereo matching for rectified images";

  m.def("compute_sparse_stereo", &compute_sparse_stereo, py::arg("left_img"),
        py::arg("right_img"), py::arg("left_points"), py::arg("templ_rows"),
        py::arg("templ_cols"), py::arg("stripe_rows"), py::arg("stripe_cols"),
        py::arg("stereo_max_y_diff"), py::arg("min_disparity"),
        py::arg("template_matching_tolerance"),
        py::arg("subpixel_refinement") = false,
        R"doc(
Compute sparse stereo matches for rectified images using
template matching along epipolar lines.

Parameters
----------
left_img : np.ndarray (H, W), uint8
    Rectified left grayscale image.
right_img : np.ndarray (H, W), uint8
    Rectified right grayscale image.
left_points : np.ndarray (N, 2), float32
    Left keypoints [x, y] in pixel coordinates.
templ_rows, templ_cols : int
    Template size in pixels (height, width).
stripe_rows, stripe_cols : int
    Stripe size in pixels (height, width).
stereo_max_y_diff : float
    Max abs(y_left - y_right) for match to be accepted.
min_disparity : float
    Minimum allowed disparity (x_left - x_right).
template_matching_tolerance : float
    Maximum allowed normalized SSD score.
subpixel_refinement : bool
    Whether to refine matches using cornerSubPix.

Returns
-------
valid_mask : np.ndarray (N,), bool
right_points : np.ndarray (N, 2), float32
scores : np.ndarray (N,), float32
)doc");
}