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

#ifndef SPIRAL_ENVIRONMENTS_LIBMYPAINT_WRAPPER_BRUSH_H_
#define SPIRAL_ENVIRONMENTS_LIBMYPAINT_WRAPPER_BRUSH_H_

#include <memory>
#include <string>

#include "libmypaint/mypaint-brush.h"
#include "spiral/environments/libmypaint_wrapper/surface.h"

namespace libmypaint_wrapper {

class BrushWrapper {
 public:
  BrushWrapper();
  ~BrushWrapper();

  void LoadFromFile(const std::string& filename);
  void SetBaseValue(int setting, float value);

  void SetSurface(SurfaceWrapper* surface) {
    surface_ = surface;
  }

  void Reset();
  void NewStroke();
  void StrokeTo(float x, float y, float pressure, float dtime);

 private:
  // Owned by BrushWrapper.
  MyPaintBrush* brush_;
  // Owned by user.
  SurfaceWrapper* surface_;
};

}  // namespace libmypaint_wrapper

#endif  // SPIRAL_ENVIRONMENTS_LIBMYPAINT_WRAPPER_BRUSH_H_
