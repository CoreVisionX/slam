#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

struct StereoSample {
    double timestamp_s;
    cv::Mat left;
    cv::Mat right;
    StereoSample() : timestamp_s(0.0) {}
    StereoSample(double ts, cv::Mat l, cv::Mat r)
        : timestamp_s(ts), left(std::move(l)), right(std::move(r)) {}
};

struct ImuFrame {
    double timestamp_s; // Seconds
    rs2_vector data;    // x, y, z
    bool is_accel;      // true=accel, false=gyro
};

struct ImuOutput {
    double timestamp;
    double gx, gy, gz;
    double ax, ay, az;
};

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

py::array_t<uint8_t> to_rgb_array(const cv::Mat &gray_mat) {
    if (gray_mat.empty()) return py::array_t<uint8_t>();
    int w = gray_mat.cols;
    int h = gray_mat.rows;
    auto *buffer = new std::vector<uint8_t>(w * h * 3);
    cv::Mat rgb_wrapper(h, w, CV_8UC3, buffer->data());
    cv::cvtColor(gray_mat, rgb_wrapper, cv::COLOR_GRAY2RGB);
    auto capsule = py::capsule(buffer, [](void *ptr) {
        delete static_cast<std::vector<uint8_t>*>(ptr);
    });
    return py::array_t<uint8_t>({h, w, 3}, buffer->data(), capsule);
}

// -----------------------------------------------------------------------------
// Main Iterator Class
// -----------------------------------------------------------------------------

class D435iIterator {
    int width_, height_, fps_;
    bool running_ = false;

    rs2::pipeline pipe_;
    rs2::config cfg_;
    rs2::pipeline_profile profile_; // active profile after start()

    std::mutex mutex_;
    std::condition_variable cv_;

    std::deque<StereoSample> frames_;
    std::deque<ImuOutput> imu_ready_queue_;

    bool is_initialized_time_base_ = false;
    double ros_time_base_ = 0.0;    // System time at start
    double camera_time_base_ = 0.0; // Hardware time at start

    std::deque<ImuFrame> imu_history_;

    bool accel0_valid_ = false;
    ImuFrame accel0_;

public:
    D435iIterator(int width = 848, int height = 480, int fps = 30)
        : width_(width), height_(height), fps_(fps) {
        start();
    }

    ~D435iIterator() { stop(); }

    void start() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            frames_.clear();
            imu_ready_queue_.clear();
            imu_history_.clear();
            accel0_valid_ = false;
            is_initialized_time_base_ = false;
            running_ = false;
            profile_ = rs2::pipeline_profile{};
        }

        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() > 0) {
            auto dev = devices[0];
            auto sensors = dev.query_sensors();
            for (auto& s : sensors) {
                // Enable motion correction to get corrected IMU data
                if (s.supports(RS2_OPTION_ENABLE_MOTION_CORRECTION))
                    s.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION, 1.f);

                // Disable Global Time to ensure we get Hardware timestamps
                if (s.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
                    s.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0.f);

                // Disable IR emitter
                if (s.supports(RS2_OPTION_EMITTER_ENABLED))
                    s.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);

                // Enable auto exposure
                if (s.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE))
                    s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);

                // Enable auto white balance
                if (s.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE))
                    s.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1.f);
            }
        }

        cfg_.disable_all_streams();
        cfg_.enable_stream(RS2_STREAM_INFRARED, 1, width_, height_, RS2_FORMAT_Y8, fps_);
        cfg_.enable_stream(RS2_STREAM_INFRARED, 2, width_, height_, RS2_FORMAT_Y8, fps_);
        cfg_.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
        cfg_.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = true;
        }

        // Capture pipeline_profile so we can query calibration for the active profile
        profile_ = pipe_.start(cfg_, [this](const rs2::frame& frame) {
            double frame_time_ms = frame.get_timestamp();
            double systetimestamp_s = 0.0;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!is_initialized_time_base_) {
                    ros_time_base_ = std::chrono::duration<double>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count();
                    camera_time_base_ = frame_time_ms;
                    is_initialized_time_base_ = true;
                }

                double elapsed_camera_ms = frame_time_ms - camera_time_base_;
                systetimestamp_s = ros_time_base_ + (elapsed_camera_ms * 1e-3);
            }

            // Dispatch
            if (auto fs = frame.as<rs2::frameset>()) {
                handle_frameset(fs, systetimestamp_s);
            } else if (auto mf = frame.as<rs2::motion_frame>()) {
                handle_motion(mf, systetimestamp_s);
            }
        });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();
        try { pipe_.stop(); } catch(...) {}
    }

    D435iIterator& iter() { return *this; }

    auto next() {
        StereoSample frame_out;
        std::vector<ImuOutput> imu_batch;

        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait for frame AND synchronized IMU data covering that frame
            cv_.wait(lock, [this] {
                if (!running_) return true;
                if (frames_.empty()) return false;
                if (imu_ready_queue_.empty()) return false;
                // Ensure IMU stream has passed the frame time
                return imu_ready_queue_.back().timestamp >= frames_.front().timestamp_s;
            });

            if (!running_) throw py::stop_iteration();

            frame_out = std::move(frames_.front());
            frames_.pop_front();

            // Extract IMU samples up to the frame time (plus small tolerance)
            while (!imu_ready_queue_.empty()) {
                if (imu_ready_queue_.front().timestamp > frame_out.timestamp_s + 0.002) {
                    break;
                }
                imu_batch.push_back(imu_ready_queue_.front());
                imu_ready_queue_.pop_front();
            }
        }

        auto left_py = to_rgb_array(frame_out.left);
        auto right_py = to_rgb_array(frame_out.right);

        size_t N = imu_batch.size();
        py::array_t<double> imu_ts({(py::ssize_t)N});
        py::array_t<double> imu_gyro({(py::ssize_t)N, (py::ssize_t)3});
        py::array_t<double> imu_acc({(py::ssize_t)N, (py::ssize_t)3});

        auto r_ts = imu_ts.mutable_unchecked<1>();
        auto r_gyro = imu_gyro.mutable_unchecked<2>();
        auto r_acc = imu_acc.mutable_unchecked<2>();

        for (size_t i = 0; i < N; ++i) {
            r_ts(i) = imu_batch[i].timestamp;

            r_gyro(i, 0) = imu_batch[i].gx;
            r_gyro(i, 1) = imu_batch[i].gy;
            r_gyro(i, 2) = imu_batch[i].gz;

            r_acc(i, 0) = imu_batch[i].ax;
            r_acc(i, 1) = imu_batch[i].ay;
            r_acc(i, 2) = imu_batch[i].az;
        }

        return std::make_tuple(frame_out.timestamp_s, left_py, right_py, imu_ts, imu_gyro, imu_acc);
    }

    // -------------------------------------------------------------------------
    // Calibration getter for the active (width,height,fps) profile
    //
    // Returns a dict:
    //   left_rect      : 3x3 K matrix for left IR intrinsics
    //   right_rect     : 3x3 K matrix for right IR intrinsics
    //   imu_from_left  : 4x4 transform (left IR cam -> IMU)
    //   imu_from_right : 4x4 transform (right IR cam -> IMU)
    // -------------------------------------------------------------------------
    py::dict get_calib_params() {
        rs2::pipeline_profile prof;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            // prefer stored profile; if absent, try active profile
            if (profile_) {
                prof = profile_;
            }
        }

        if (!prof) {
            try {
                prof = pipe_.get_active_profile();
            } catch (...) {
                throw std::runtime_error("Pipeline not started; cannot query calibration.");
            }
        }

        auto left_sp  = prof.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
        auto right_sp = prof.get_stream(RS2_STREAM_INFRARED, 2).as<rs2::video_stream_profile>();

        // Prefer gyro as IMU frame; fall back to accel if needed.
        rs2::stream_profile imu_sp;
        try {
            imu_sp = prof.get_stream(RS2_STREAM_GYRO);
        } catch (...) {
            try {
                imu_sp = prof.get_stream(RS2_STREAM_ACCEL);
            } catch (...) {
                throw std::runtime_error("No IMU stream (gyro/accel) found in active profile.");
            }
        }

        auto K_from_intr = [](const rs2_intrinsics& in) {
            py::array_t<double> K({3, 3});
            auto k = K.mutable_unchecked<2>();
            k(0,0) = in.fx;  k(0,1) = 0.0;   k(0,2) = in.ppx;
            k(1,0) = 0.0;    k(1,1) = in.fy; k(1,2) = in.ppy;
            k(2,0) = 0.0;    k(2,1) = 0.0;   k(2,2) = 1.0;
            return K;
        };

        rs2_intrinsics li = left_sp.get_intrinsics();
        rs2_intrinsics ri = right_sp.get_intrinsics();

        py::array_t<double> left_rect  = K_from_intr(li);
        py::array_t<double> right_rect = K_from_intr(ri);

        auto T_from_extr = [](const rs2_extrinsics& ex) {
            py::array_t<double> T({4, 4});
            auto t = T.mutable_unchecked<2>();

            // rs2_extrinsics.rotation is column-major 3x3.
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    t(r, c) = static_cast<double>(ex.rotation[c * 3 + r]);
                }
            }

            t(0,3) = static_cast<double>(ex.translation[0]);
            t(1,3) = static_cast<double>(ex.translation[1]);
            t(2,3) = static_cast<double>(ex.translation[2]);

            t(3,0) = 0.0; t(3,1) = 0.0; t(3,2) = 0.0; t(3,3) = 1.0;
            return T;
        };

        rs2_extrinsics ex_imu_from_left  = left_sp.get_extrinsics_to(imu_sp);
        rs2_extrinsics ex_imu_from_right = right_sp.get_extrinsics_to(imu_sp);

        py::array_t<double> imu_from_left  = T_from_extr(ex_imu_from_left);
        py::array_t<double> imu_from_right = T_from_extr(ex_imu_from_right);

        double baseline = std::abs(ex_imu_from_left.translation[0] - ex_imu_from_right.translation[0]);

        py::dict out;
        out["K_left_rect"] = left_rect;
        out["K_right_rect"] = right_rect;
        out["imu_from_left"] = imu_from_left;
        out["imu_from_right"] = imu_from_right;
        out["baseline"] = baseline;
        return out;
    }

private:
    void handle_frameset(const rs2::frameset& fs, double systetimestamp_s) {
        rs2::video_frame left = fs.get_infrared_frame(1);
        rs2::video_frame right = fs.get_infrared_frame(2);

        // RealSense frameset parts share the same timestamp
        cv::Mat l(left.get_height(), left.get_width(), CV_8UC1, (void*)left.get_data());
        cv::Mat r(right.get_height(), right.get_width(), CV_8UC1, (void*)right.get_data());

        std::lock_guard<std::mutex> lock(mutex_);
        frames_.emplace_back(systetimestamp_s, l.clone(), r.clone());
        cv_.notify_one();
    }

    void handle_motion(const rs2::motion_frame& f, double systetimestamp_s) {
        bool is_accel = (f.get_profile().stream_type() == RS2_STREAM_ACCEL);
        rs2_vector data = f.get_motion_data();

        std::lock_guard<std::mutex> lock(mutex_);

        // Push raw data to history
        imu_history_.push_back({systetimestamp_s, data, is_accel});

        if (imu_history_.size() < 3) return;

        while (imu_history_.size() >= 2) { // Need at least potential accel0 and accel1 or gyro
            ImuFrame& front = imu_history_.front();

            if (!accel0_valid_) {
                if (front.is_accel) {
                    accel0_ = front;
                    accel0_valid_ = true;
                }
                imu_history_.pop_front();
                continue;
            }

            int next_accel_idx = -1;
            for (size_t i = 0; i < imu_history_.size(); ++i) {
                if (imu_history_[i].is_accel) {
                    next_accel_idx = (int)i;
                    break;
                }
            }

            if (next_accel_idx == -1) {
                break;
            }

            ImuFrame accel1 = imu_history_[(size_t)next_accel_idx];

            double dt = accel1.timestamp_s - accel0_.timestamp_s;
            if (dt <= 0) {
                accel0_ = accel1;
                for (int k = 0; k <= next_accel_idx; k++) imu_history_.pop_front();
                continue;
            }

            for (int i = 0; i < next_accel_idx; ++i) {
                ImuFrame& g = imu_history_[(size_t)i];
                if (!g.is_accel) {
                    // Linear interpolation
                    double alpha = (g.timestamp_s - accel0_.timestamp_s) / dt;

                    rs2_vector interp_acc;
                    interp_acc.x = accel0_.data.x * (1.0 - alpha) + accel1.data.x * alpha;
                    interp_acc.y = accel0_.data.y * (1.0 - alpha) + accel1.data.y * alpha;
                    interp_acc.z = accel0_.data.z * (1.0 - alpha) + accel1.data.z * alpha;

                    imu_ready_queue_.push_back({
                        g.timestamp_s,
                        g.data.x, g.data.y, g.data.z,          // Gyro
                        interp_acc.x, interp_acc.y, interp_acc.z // Interpolated Accel
                    });
                }
            }

            accel0_ = accel1;
            for (int k = 0; k <= next_accel_idx; k++) imu_history_.pop_front();
        }

        cv_.notify_one();
    }
};

PYBIND11_MODULE(rs_sdk, m) {
    py::class_<D435iIterator>(m, "D435iIterator")
        .def(py::init<int, int, int>(), py::arg("width")=848, py::arg("height")=480, py::arg("fps")=30)
        .def("__iter__", &D435iIterator::iter)
        .def("__next__", &D435iIterator::next)
        .def("get_calib_params", &D435iIterator::get_calib_params)
        .def("close", &D435iIterator::stop);
}