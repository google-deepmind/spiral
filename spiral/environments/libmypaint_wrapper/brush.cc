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

#include "spiral/environments/libmypaint_wrapper/brush.h"

#include <fstream>
#include <sstream>
#include <string>

#include "glog/logging.h"

namespace libmypaint_wrapper {

BrushWrapper::BrushWrapper() : brush_(mypaint_brush_new()) {}

BrushWrapper::~BrushWrapper() {
  mypaint_brush_unref(brush_);
}

void BrushWrapper::LoadFromFile(const std::string& filename) {
  std::stringstream str_stream;
  str_stream << std::ifstream(filename).rdbuf();
  std::string brush_json = str_stream.str();
  mypaint_brush_from_string(brush_, brush_json.c_str());
}

void BrushWrapper::SetBaseValue(int setting, float value) {
  mypaint_brush_set_base_value(
      brush_, static_cast<MyPaintBrushSetting>(setting), value);
}

void BrushWrapper::Reset() {
  mypaint_brush_reset(brush_);
}

void BrushWrapper::NewStroke() {
  mypaint_brush_new_stroke(brush_);
}

void BrushWrapper::StrokeTo(float x, float y, float pressure, float dtime) {
  CHECK(surface_ != nullptr) << "Surface for the brush has not been set";

  // TODO: Consider passing tilts as parameters.
  const float xtilt = 0.0;
  const float ytilt = 0.0;
  mypaint_brush_stroke_to(
      brush_, surface_->GetInterface(), x, y, pressure, xtilt, ytilt, dtime);
}

}  // namespace libmypaint_wrapper
