#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
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

// Batch detection function
std::vector<py::dict>
batch_detect_charuco(const std::vector<py::array_t<uint8_t>> &images,
                     int squares_x, int squares_y, float square_length,
                     float marker_length, const std::string &dict_name) {
  std::vector<py::dict> results;

  // Setup dictionary and board
  cv::Ptr<cv::aruco::Dictionary> dictionary;

  // Map string name to opencv constant (simplified subset)
  if (dict_name == "DICT_6X6_250")
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  else if (dict_name == "DICT_4X4_50")
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  else
    dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250); // Fallback

  cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
      squares_x, squares_y, square_length, marker_length, dictionary);

  cv::Ptr<cv::aruco::DetectorParameters> params =
      cv::aruco::DetectorParameters::create();

  // Process images
  for (size_t i = 0; i < images.size(); ++i) {
    // We need to cast away constness for numpy_to_mat because pybind11 buffer
    // request might write (though we don't)
    py::array_t<uint8_t> img = images[i];
    cv::Mat image = numpy_to_mat(img);

    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;

    // 1. Detect Markers
    cv::aruco::detectMarkers(image, dictionary, marker_corners, marker_ids,
                             params);

    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    bool success = false;

    // 2. Interpolate Charuco Corners
    if (marker_ids.size() > 0) {
      cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, image,
                                           board, charuco_corners, charuco_ids);
      success = (charuco_corners.size() >= 4);
    }

    // Convert to python dict
    py::dict res;
    res["success"] = success;

    if (success) {
      // Convert vectors to numpy
      res["corners"] = py::array(py::cast(charuco_corners));
      res["ids"] = py::array(py::cast(charuco_ids));
    } else {
      res["corners"] = py::none();
      res["ids"] = py::none();
    }

    results.push_back(res);
  }

  return results;
}

PYBIND11_MODULE(calibration_cpp, m) {
  m.doc() = "Optimized calibration routines implemented in C++";

  m.def("batch_detect_charuco", &batch_detect_charuco,
        "Detect ChArUco board in a batch of images", py::arg("images"),
        py::arg("squares_x"), py::arg("squares_y"), py::arg("square_length"),
        py::arg("marker_length"), py::arg("dict_name"));
}
