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

#include "third_party/pybind11/include/pybind11/numpy.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/pytypes.h"
#include "third_party/pybind11/include/pybind11/stl.h"

namespace py = pybind11;

#include "spiral/environments/libmypaint_wrapper/brush.h"
#include "spiral/environments/libmypaint_wrapper/surface.h"

namespace libmypaint_wrapper {

PYBIND11_MODULE(pylibmypaint, m) {
  py::class_<BrushWrapper> brush_wrapper(m, "BrushWrapper");
  brush_wrapper
      .def(py::init())
      .def("LoadFromFile", &BrushWrapper::LoadFromFile,
            py::call_guard<py::gil_scoped_release>())
      .def("SetBaseValue", &BrushWrapper::SetBaseValue,
            py::call_guard<py::gil_scoped_release>())
      .def("SetSurface", &BrushWrapper::SetSurface,
            py::call_guard<py::gil_scoped_release>())
      .def("Reset", &BrushWrapper::Reset,
            py::call_guard<py::gil_scoped_release>())
      .def("NewStroke", &BrushWrapper::NewStroke,
            py::call_guard<py::gil_scoped_release>())
      .def("StrokeTo", &BrushWrapper::StrokeTo,
            py::call_guard<py::gil_scoped_release>());

  py::class_<SurfaceWrapper> surface_wrapper(m, "SurfaceWrapper");
  surface_wrapper
      .def(py::init<int, int, SurfaceWrapper::Background>())
      .def("BeginAtomic", &SurfaceWrapper::BeginAtomic,
           py::call_guard<py::gil_scoped_release>())
      .def("EndAtomic", &SurfaceWrapper::EndAtomic,
           py::call_guard<py::gil_scoped_release>())
      .def("Clear", &SurfaceWrapper::Clear,
           py::call_guard<py::gil_scoped_release>())
      .def("BufferAsNumpy", [](SurfaceWrapper &sw) {
            uint16_t* buffer = sw.GetBuffer();
            auto buffer_dims = sw.GetBufferDims();
            return py::array_t<uint16_t>(buffer_dims, buffer);
          });

  py::enum_<SurfaceWrapper::Background>(surface_wrapper, "Background")
      .value("kWhite", SurfaceWrapper::Background::kWhite)
      .value("kBlack", SurfaceWrapper::Background::kBlack)
      .export_values();
}

}  // namespace libmypaint_wrapper
