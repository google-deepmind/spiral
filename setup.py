# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install script for setuptools."""

import os
import subprocess
import sys

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):

  def __init__(self, name, cmake_lists_dir=".", **kwargs):
    Extension.__init__(self, name, sources=[], **kwargs)
    self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuildExt(build_ext):
  """Build extension handling building C/C++ files with CMake."""

  def run(self):
    try:
      subprocess.check_output(["cmake", "--version"])
    except OSError:
      raise RuntimeError("Cannot find CMake executable")

    for ext in self.extensions:
      self.build_extension(ext)

  def build_extension(self, ext):
    self.configure_cmake(ext)
    self.build_cmake(ext)

  def configure_cmake(self, ext):
    extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
    cfg = "Debug" if self.debug else "Release"

    configure_cmd = ["cmake", ext.cmake_lists_dir]

    configure_cmd += [
        "-DCMAKE_BUILD_TYPE={}".format(cfg),
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}/spiral/environments".format(
            cfg.upper(), extdir),
        "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
            cfg.upper(), self.build_temp),
        "-DPYTHON_EXECUTABLE:FILEPATH={}".format(sys.executable),
    ]

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    subprocess.check_call(configure_cmd, cwd=self.build_temp)

  def build_cmake(self, ext):
    cfg = "Debug" if self.debug else "Release"
    subprocess.check_call(["cmake", "--build", ".", "--config", cfg],
                          cwd=self.build_temp)


spiral_extension = CMakeExtension("spiral")

setup(
    name="spiral",
    version="1.0",
    author="DeepMind",
    license="Apache License, Version 2.0",
    packages=find_packages(include=["spiral*"]),
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=[
        "tensorflow>=1.14,<2",
        "tensorflow-hub",
        "dm-sonnet>=1.35,<2",
        "dm-env",
        "six",
        "scipy",
        "numpy",
    ],
    ext_modules=[spiral_extension],
    cmdclass={
        "build_ext": CMakeBuildExt,
    },
)
