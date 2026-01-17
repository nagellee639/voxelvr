#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

// Helper to convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t> &input) {
  py::buffer_info buf = input.request();
  if (buf.ndim != 3 && buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be 2 or 3");
  }

  int rows = buf.shape[0];
  int cols = buf.shape[1];
  int type = (buf.ndim == 3) ? CV_8UC3 : CV_8UC1;

  return cv::Mat(rows, cols, type, (void *)buf.ptr);
}

// Structure to hold detection results
struct DetectionResult {
  bool success;
  std::vector<std::vector<cv::Point2f>>
      corners; // Not used for charuco return, but legacy
  std::vector<int> ids;
  std::vector<cv::Point2f> charuco_corners;
  std::vector<int> charuco_ids;
};

// Thread-safe progress reporter
class ProgressReporter {
public:
  ProgressReporter(py::function callback, int total)
      : callback_(callback), total_(total), current_(0) {}

  void update() {
    int c = ++current_;
    // Acquire GIL only when calling back to Python
    // We can throttle this if needed, but for now calling every frame
    // is okay if it's not too frequent. To be safe, let's just call it.
    // Ideally we might want to do this every X frames.
    if (callback_) {
      py::gil_scoped_acquire acquire;
      try {
        callback_(c, total_);
      } catch (...) {
        // Ignore errors in callback to prevent crashing C++
      }
    }
  }

private:
  py::function callback_;
  int total_;
  std::atomic<int> current_;
};

// Single image processing function
DetectionResult
process_single_image(const cv::Mat &image,
                     cv::Ptr<cv::aruco::Dictionary> dictionary,
                     cv::Ptr<cv::aruco::CharucoBoard> board,
                     cv::Ptr<cv::aruco::DetectorParameters> params) {
  DetectionResult result;
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;

  // 1. Detect Markers
  cv::aruco::detectMarkers(image, dictionary, marker_corners, marker_ids,
                           params);

  // 2. Interpolate Charuco Corners
  if (marker_ids.size() > 0) {
    cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, image,
                                         board, result.charuco_corners,
                                         result.charuco_ids);
    result.success = (result.charuco_corners.size() >= 4);
  } else {
    result.success = false;
  }
  return result;
}

std::vector<py::dict> batch_detect_charuco(py::object images_obj, int squares_x,
                                           int squares_y, float square_length,
                                           float marker_length,
                                           const std::string &dict_name,
                                           py::object progress_callback) {

  // 1. Setup OpenCV objects (with GIL, fast)
  cv::Ptr<cv::aruco::Dictionary> dictionary;
  if (dict_name == "DICT_6X6_250")
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  else if (dict_name == "DICT_4X4_50")
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  else
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

  cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
      squares_x, squares_y, square_length, marker_length, dictionary);
  cv::Ptr<cv::aruco::DetectorParameters> params =
      cv::aruco::DetectorParameters::create();

  // 2. Clone images to C++ memory (with GIL)
  std::vector<cv::Mat> mat_images;
  py::list images = images_obj.cast<py::list>();
  mat_images.reserve(images.size());
  for (auto item : images) {
    py::array_t<uint8_t> img = item.cast<py::array_t<uint8_t>>();
    cv::Mat m = numpy_to_mat(const_cast<py::array_t<uint8_t> &>(img));
    mat_images.push_back(m.clone());
  }

  // 3. Prepare for parallel execution
  size_t count = mat_images.size();
  std::vector<DetectionResult> cpp_results(count);
  std::atomic<int> completed{0};

  {
    py::gil_scoped_release release;

    // Use hardware concurrency
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0)
      n_threads = 4;

    // Simple parallel for loop using chunks or just async
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < count; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        cpp_results[i] =
            process_single_image(mat_images[i], dictionary, board, params);
        int c = ++completed;

        // Callback to update progress
        // We need to re-acquire GIL to call Python function
        if (!progress_callback.is_none()) {
          py::gil_scoped_acquire acquire;
          try {
            progress_callback(c, (int)count);
          } catch (...) {
          } // Suppress python errors during callback
        }
      }));
    }

    // Wait for all to finish
    for (auto &f : futures) {
      f.wait();
    }
  } // GIL re-acquired

  // 4. Convert results to Python objects (with GIL)
  std::vector<py::dict> py_results;
  py_results.reserve(count);

  for (const auto &res : cpp_results) {
    py::dict py_res;
    py_res["success"] = res.success;

    if (res.success) {
      size_t n_points = res.charuco_corners.size();
      py::array_t<float> corners_arr({(long)n_points, 2L});
      py::array_t<int> ids_arr({(long)n_points, 1L});

      auto corners_ptr = corners_arr.mutable_unchecked<2>();
      auto ids_ptr = ids_arr.mutable_unchecked<2>();

      for (size_t k = 0; k < n_points; ++k) {
        corners_ptr(k, 0) = res.charuco_corners[k].x;
        corners_ptr(k, 1) = res.charuco_corners[k].y;
        ids_ptr(k, 0) = res.charuco_ids[k];
      }
      py_res["corners"] = corners_arr;
      py_res["ids"] = ids_arr;
    } else {
      py_res["corners"] = py::none();
      py_res["ids"] = py::none();
    }
    py_results.push_back(py_res);
  }

  return py_results;
}

PYBIND11_MODULE(calibration_cpp_v2, m) {
  m.doc() = "Optimized calibration routines implemented in C++";

  m.def("batch_detect_charuco", &batch_detect_charuco,
        "Detect ChArUco board in a batch of images", py::arg("images"),
        py::arg("squares_x"), py::arg("squares_y"), py::arg("square_length"),
        py::arg("marker_length"), py::arg("dict_name"),
        py::arg("progress_callback") = py::none());
}
