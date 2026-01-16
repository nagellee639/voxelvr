
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the ``get_include()`` method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

# Add /usr/include/opencv4 to include_dirs based on findings
ext_modules = [
    Extension(
        'voxelvr.calibration.calibration_cpp',
        ['voxelvr/calibration/cpp/optimization.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/opencv4', 
        ],
        libraries=['opencv_core', 'opencv_aruco', 'opencv_calib3d', 'opencv_imgproc', 'opencv_imgcodecs'],
        language='c++'
    ),
]

setup(
    name='voxelvr.calibration.calibration_cpp',
    version='0.1',
    author='VoxelVR Team',
    description='Optimized calibration routines',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.10'],
    install_requires=['pybind11>=2.10'],
)
