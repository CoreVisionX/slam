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
    double timestamp_s;     // Seconds
    rs2_vector data;   // x, y, z
    bool is_accel;       // true=accel, false=gyro
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
        frames_.clear();
        imu_ready_queue_.clear();
        imu_history_.clear();
        accel0_valid_ = false;
        is_initialized_time_base_ = false;

        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() > 0) {
            auto dev = devices[0];
            auto sensors = dev.query_sensors();
            for (auto& s : sensors) {
                // Disable motion correction to get raw data
                if (s.supports(RS2_OPTION_ENABLE_MOTION_CORRECTION))
                    s.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION, 0);
                
                // Disable Global Time to ensure we get Hardware timestamps
                if (s.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
                    s.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0);

                // Disable IR emitter
                if (s.supports(RS2_OPTION_EMITTER_ENABLED))
                    s.set_option(RS2_OPTION_EMITTER_ENABLED, 0);
            }
        }

        cfg_.enable_stream(RS2_STREAM_INFRARED, 1, width_, height_, RS2_FORMAT_Y8, fps_);
        cfg_.enable_stream(RS2_STREAM_INFRARED, 2, width_, height_, RS2_FORMAT_Y8, fps_);
        cfg_.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
        cfg_.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

        running_ = true;

        pipe_.start(cfg_, [this](const rs2::frame& frame) {
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

            // 2. Dispatch
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

private:
    void handle_frameset(const rs2::frameset& fs, double systetimestamp_s) {
        rs2::video_frame left = fs.get_infrared_frame(1);
        rs2::video_frame right = fs.get_infrared_frame(2);

        // Note: RealSense frameset parts share the same timestamp
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
            for(size_t i = 0; i < imu_history_.size(); ++i) {
                if (imu_history_[i].is_accel) {
                    next_accel_idx = i;
                    break;
                }
            }

            if (next_accel_idx == -1) {
                break; 
            }

            ImuFrame accel1 = imu_history_[next_accel_idx];
            
            double dt = accel1.timestamp_s - accel0_.timestamp_s;
            if (dt <= 0) {
                // Should not happen with monotonic clocks, but safe guard
                accel0_ = accel1;
                // Remove everything up to and including this bad packet
                for(int k=0; k<=next_accel_idx; k++) imu_history_.pop_front();
                continue;
            }

            for(int i = 0; i < next_accel_idx; ++i) {
                ImuFrame& g = imu_history_[i];
                if (!g.is_accel) {
                    // Linear Interpolation
                    double alpha = (g.timestamp_s - accel0_.timestamp_s) / dt;
                    
                    rs2_vector interp_acc;
                    interp_acc.x = accel0_.data.x * (1.0 - alpha) + accel1.data.x * alpha;
                    interp_acc.y = accel0_.data.y * (1.0 - alpha) + accel1.data.y * alpha;
                    interp_acc.z = accel0_.data.z * (1.0 - alpha) + accel1.data.z * alpha;

                    // Publish
                    imu_ready_queue_.push_back({
                        g.timestamp_s,
                        g.data.x, g.data.y, g.data.z, // Gyro
                        interp_acc.x, interp_acc.y, interp_acc.z // Interpolated Accel
                    });
                }
            }

            accel0_ = accel1;

            for(int k=0; k<=next_accel_idx; k++) imu_history_.pop_front();
        }
        
        cv_.notify_one();
    }
};

PYBIND11_MODULE(rs_sdk, m) {
    py::class_<D435iIterator>(m, "D435iIterator")
        .def(py::init<int, int, int>(), py::arg("width")=848, py::arg("height")=480, py::arg("fps")=30)
        .def("__iter__", &D435iIterator::iter)
        .def("__next__", &D435iIterator::next)
        .def("close", &D435iIterator::stop);
}
