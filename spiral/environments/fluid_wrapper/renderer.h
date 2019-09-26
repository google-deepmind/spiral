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

#ifndef SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_RENDERER_H_
#define SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_RENDERER_H_

#include <map>

#include "third_party/swiftshader/include/GLES3/gl3.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "spiral/environments/fluid_wrapper/brush.h"
#include "config.pb.h"
#include "spiral/environments/fluid_wrapper/simulator.h"
#include "spiral/environments/fluid_wrapper/utils.h"

namespace spiral {
namespace fluid {

class Renderer {
 public:
  explicit Renderer(const Config& config);
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  void Setup(int width, int height, absl::string_view shader_base_dir);
  void SetBrushColor(float h, float s, float v, float a);
  void Update(float x, float y, float scale, bool is_painting);
  void Render();
  void Reset();
  void Cleanup();

  unsigned char * GetCanvas() {
    return &canvas_[0];
  }

 private:
  enum class ProgramId {
      kPainting, kOutput, kTest
  };

  enum class TextureId {
      kCanvas
  };

  void ClearCanvas();
  void RunPainting(Rectangle<float> painting_rectangle);;

  // Constant settings.
  const float kBrushHeight = 2.0;
  const float kZThreshold = 0.13333;

  const float kSplatVelocityScale = 0.14;
  const float kSplatRadius = 0.05;

  const float kMinAlpha = 0.002;
  const float kMaxAlpha = 0.025;

  const float kNormalScale = 7.0;
  const float kRoughness = 0.075;
  const float kF0 = 0.05;
  const float kSpecularScale = 0.5;
  const float kDiffuseScale = 0.15;
  const float kLightDirection[3] = {0.0f, 1.0f, 1.0f};

  // Non-constant variables.
  int width_ = 0;
  int height_ = 0;
  bool canvas_updated_ = false;

  Simulator simulator_;
  Brush brush_;

  std::vector<float> brush_color_;

  std::vector<unsigned char> canvas_;

  // OpenGL stuff.
  GLuint vao_ = 0;

  typedef absl::flat_hash_map<std::string, GLuint> StrToLocMap;

  absl::flat_hash_map<ProgramId, GLuint> program_map_;
  absl::flat_hash_map<ProgramId, StrToLocMap> attrib_loc_map_;
  absl::flat_hash_map<ProgramId, StrToLocMap> uniform_loc_map_;

  absl::flat_hash_map<TextureId, GLuint> texture_map_;

  GLuint quad_vbo_ = 0;
  GLuint framebuffer_;
};

}  // namespace fluid
}  // namespace spiral

#endif  // SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_RENDERER_H_
