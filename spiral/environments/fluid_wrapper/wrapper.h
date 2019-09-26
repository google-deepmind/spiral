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


#ifndef SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_WRAPPER_H_
#define SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_WRAPPER_H_

#include <cstdint>
#include <vector>

#include "spiral/environments/fluid_wrapper/egl.h"
#include "config.pb.h"
#include "spiral/environments/fluid_wrapper/renderer.h"

namespace spiral {
namespace fluid {

class Wrapper {
 public:
  explicit Wrapper(const Config& config);
  ~Wrapper();

  Wrapper(const Wrapper&) = delete;
  Wrapper& operator=(const Wrapper&) = delete;

  void Setup(int width, int height, const std::string& shader_base_dir);

  void Reset() {
    renderer_.Reset();
  }

  void Update(float x, float y, float scale, bool is_painting) {
    renderer_.Update(x, y, scale, is_painting);
  }

  void SetBrushColor(float h, float s, float v, float a) {
    renderer_.SetBrushColor(h, s, v, a);
  }

  uint8_t* GetCanvas();
  std::vector<int> GetCanvasDims() const;

 private:
  int width_;
  int height_;

  Renderer renderer_;

  EGLDisplay egl_display_;
  EGLSurface egl_surface_;
  EGLContext egl_context_;
  bool is_setup_;
};

}  // namespace fluid
}  // namespace spiral

#endif  // SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_WRAPPER_H_
