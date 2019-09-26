// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>

#include "third_party/pybind11/include/pybind11/numpy.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/pytypes.h"
#include "third_party/pybind11/include/pybind11/stl.h"

namespace py = pybind11;

#include "config.pb.h"
#include "spiral/environments/fluid_wrapper/wrapper.h"

namespace spiral {
namespace fluid {

PYBIND11_MODULE(pyfluid, m) {
  py::class_<Wrapper> wrapper(m, "Wrapper");
  wrapper
      .def(py::init([](std::string serialized_config) {
            Config config;
            config.ParseFromString(serialized_config);
            return std::unique_ptr<Wrapper>(new Wrapper(config));
          }))
      .def("Setup", &Wrapper::Setup,
            py::call_guard<py::gil_scoped_release>())
      .def("Reset", &Wrapper::Reset,
            py::call_guard<py::gil_scoped_release>())
      .def("Update", &Wrapper::Update,
            py::call_guard<py::gil_scoped_release>())
      .def("SetBrushColor", &Wrapper::SetBrushColor,
            py::call_guard<py::gil_scoped_release>())
      .def("CanvasAsNumpy", [](Wrapper &w) {
            uint8_t* canvas = w.GetCanvas();
            auto canvas_dims = w.GetCanvasDims();
            return py::array_t<uint8_t>(canvas_dims, canvas);
          });
}

}  // namespace fluid
}  // namespace spiral
