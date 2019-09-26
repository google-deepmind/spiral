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

#ifndef SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_SIMULATOR_H_
#define SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_SIMULATOR_H_

#include <list>
#include <map>

#include "third_party/swiftshader/include/GLES3/gl3.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "spiral/environments/fluid_wrapper/brush.h"
#include "config.pb.h"
#include "spiral/environments/fluid_wrapper/utils.h"

namespace spiral {
namespace fluid {

class Simulator {
 public:
  enum class Status {
      kSkippedSimulation, kFinishedSimulation
  };

  enum class TextureId {
      kPaint, kPaintTemp, kVelocity, kVelocityTemp, kDivergence,
      kPressure, kPressureTemp
  };

  explicit Simulator(const Config::Simulator& config);
  Simulator(const Simulator&) = delete;
  Simulator& operator=(const Simulator&) = delete;

  void Setup(int width, int height, absl::string_view shader_base_dir);
  void Splat(const Brush &brush,
             Rectangle<float> painting_rectangle,
             float z_threshold,
             float splat_color[4],
             float splat_radius,
             float velocity_scale);
  Status Simulate();
  void Reset();

  int width() const {
    return width_;
  }

  int height() const {
    return height_;
  }

  GLuint GetTexture(TextureId texture_id) const {
    return texture_map_.at(texture_id);
  }

 private:
  enum class ProgramId {
      kSplat, kVelocitySplat, kAdvect, kDivergence, kJacobi, kSubtract
  };

  void ClearTextures(std::vector<TextureId> texture_ids);
  Rectangle<float> GetSimulationArea();
  void CleanupSplatAreas();

  void RunAdvect(Rectangle<float> simulation_area,
                 TextureId velocity_texture,
                 TextureId data_texture,
                 TextureId target_texture,
                 float delta_time,
                 float dissipation);
  void RunDivergence(Rectangle<float> simulation_area);
  void RunJacobi(Rectangle<float> simulation_area);
  void RunSubtract(Rectangle<float> simulation_area);
  void RunSplat(const Brush &brush,
                Rectangle<float> simulation_area,
                Rectangle<float> painting_rectangle,
                float radius,
                float color[4],
                float z_threshold);
  void RunVelocitySplat(const Brush &brush,
                        Rectangle<float> simulation_area,
                        Rectangle<float> painting_rectangle,
                        float radius,
                        float z_threshold,
                        float velocity_scale);

  // Constant settings.
  const unsigned int kNumJacobiIterations = 2;

  const unsigned int kNumFramesToSimulate = 60;

  const float kSplatPadding = 4.5;
  const float kSpeedPadding = 1.1;

  // Non-constant variables.
  int width_ = 0;
  int height_ = 0;
  int frame_number_ = 0;
  std::list<std::pair<Rectangle<float>, int>> splat_areas_;

  typedef absl::flat_hash_map<std::string, GLuint> StrToLocMap;

  absl::flat_hash_map<ProgramId, GLuint> program_map_;
  absl::flat_hash_map<ProgramId, StrToLocMap> attrib_loc_map_;
  absl::flat_hash_map<ProgramId, StrToLocMap> uniform_loc_map_;

  absl::flat_hash_map<TextureId, GLuint> texture_map_;

  GLuint quad_vbo_ = 0;
  GLuint simulation_framebuffer_ = 0;
};

}  // namespace fluid
}  // namespace spiral

#endif  // SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_SIMULATOR_H_
