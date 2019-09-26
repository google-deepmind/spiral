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

# Libmypaint wrapper target.
set(SOURCES
    environments/libmypaint_wrapper/python/pylibmypaint.cc
    environments/libmypaint_wrapper/brush.cc
    environments/libmypaint_wrapper/surface.cc)

pybind11_add_module(pylibmypaint ${SOURCES})
add_dependencies(pylibmypaint libmypaint)

target_include_directories(pylibmypaint PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(pylibmypaint PUBLIC ${Libmypaint_INCLUDE_DIRS})
target_include_directories(pylibmypaint PUBLIC ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(
    pylibmypaint PRIVATE
    absl::memory glog::glog ${Libmypaint_LIBRARIES} ${Libmypaint_DEPS})

# Fluid Paint wrapper target.
protobuf_generate_cpp(
    PROTO_SOURCES PROTO_HEADERS
    environments/fluid_wrapper/protos/config.proto)
protobuf_generate_python(
    PROTO_PY
    environments/fluid_wrapper/protos/config.proto)

set(SOURCES
    environments/fluid_wrapper/python/pyfluid.cc
    environments/fluid_wrapper/brush.cc
    environments/fluid_wrapper/error.cc
    environments/fluid_wrapper/renderer.cc
    environments/fluid_wrapper/simulator.cc
    environments/fluid_wrapper/utils.cc
    environments/fluid_wrapper/wrapper.cc
    ${PROTO_SOURCES}
    ${PROTO_HEADERS})

set(SWIFTSHADER_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/third_party/swiftshader/include)

pybind11_add_module(pyfluid ${SOURCES})

target_include_directories(pyfluid PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(pyfluid PUBLIC ${SWIFTSHADER_INCLUDE_DIRS})
target_include_directories(pyfluid PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(pyfluid PUBLIC ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(
    pyfluid PRIVATE
    absl::memory absl::flat_hash_map glog::glog protobuf::libprotobuf
    libEGL libGLESv2 SwiftShader)

add_custom_target(
    pyconfig
    ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${PROTO_PY} ${PROJECT_SOURCE_DIR}/spiral/environments
    DEPENDS ${PROTO_PY})
