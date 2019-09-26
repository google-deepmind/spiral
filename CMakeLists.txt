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

cmake_minimum_required(VERSION 3.14)
project(spiral)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(PythonInterp ${Python3_VERSION} REQUIRED)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# Build libmypaint.
include(ExternalProject)
set(LIBMYPAINT_CFLAGS "")
ExternalProject_Add(
  libmypaint
  URL https://github.com/mypaint/libmypaint/releases/download/v1.3.0/libmypaint-1.3.0.tar.xz
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --disable-static --enable-shared ${LIBMYPAINT_CFLAGS}
  BUILD_COMMAND ${MAKE} ${LIBMYPAINT_CFLAGS}
  BUILD_IN_SOURCE 1
)
ExternalProject_Get_property(libmypaint INSTALL_DIR)
set(Libmypaint_INCLUDE_DIRS ${INSTALL_DIR}/include ${INSTALL_DIR}/include/libmypaint)
set(Libmypaint_LIBRARIES ${INSTALL_DIR}/lib/libmypaint.so)

find_library(JSONC_LIBRARY NAMES json-c)
set(Libmypaint_DEPS ${JSONC_LIBRARY})

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/absl)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/glog)

set(BUILD_VULKAN OFF CACHE BOOL "Build the Vulkan library")
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/swiftshader)

set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)

add_subdirectory(spiral)
