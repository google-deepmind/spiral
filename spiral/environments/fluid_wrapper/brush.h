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

#ifndef SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_BRUSH_H_
#define SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_BRUSH_H_

#include <algorithm>
#include <csignal>
#include <deque>
#include <map>

#include "third_party/swiftshader/include/GLES3/gl3.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "config.pb.h"
#include "spiral/environments/fluid_wrapper/utils.h"

namespace spiral {
namespace fluid {

class Brush {
 public:
  enum class VboId {
      kSplatCoords, kSplatIndices, kBrushTextureCoords, kBrushIndices
  };

  enum class TextureId {
      kPositions, kPreviousPositions, kVelocities, kPreviousVelocities,
      kProjectedPositions, kProjectedPositionsTemp, kRandoms
  };

  explicit Brush(const Config::Brush& config);
  Brush(const Brush&) = delete;
  Brush& operator=(const Brush&) = delete;

  void Setup(absl::string_view shader_base_dir);
  void Initialize(float x, float y, float z, float scale);
  void Update(float x, float y, float z, float scale);
  void Reset();

  bool IsInitialized() const {
    return is_initialized_;
  }

  float x() const {
    return position_x_;
  }

  float y() const {
    return position_y_;
  }

  float z() const {
    return position_z_;
  }

  float Scale() const {
    return scale_;
  }

  float BristleCount() const {
    return kNumBristles;
  }

  float GetFilteredSpeed() const {
    return *std::max_element(speeds_.begin(), speeds_.end());
  }

  GLuint GetVbo(VboId vbo_id) const {
    return vbo_map_.at(vbo_id);
  }

  GLuint GetTexture(TextureId texture_id) const {
    return texture_map_.at(texture_id);
  }

  int NumSplatIndicesToRender() const {
    return num_splat_indices_;
  }

 private:
  enum class ProgramId {
      kProject, kDistanceConstraint, kPlaneConstraint, kBendingConstraint,
      kSetBristles, kUpdateVelocity
  };

  void ClearTextures(std::vector<TextureId> texture_ids);
  void RunSetBristles(Rectangle<GLint> simulation_area,
                      TextureId target_texture);
  void RunProject(Rectangle<GLint> simulation_area);
  void RunDistanceConstraint(Rectangle<GLint> simulation_area, int pass);
  void RunBendingConstraint(Rectangle<GLint> simulation_area, int pass);
  void RunPlaneConstraint(Rectangle<GLint> simulation_area);
  void RunUpdateVelocity(Rectangle<GLint> simulation_area);

  // Constant settings.
  const unsigned int kNumPreviousSpeeds = 15;

  const unsigned int kNumSplatsPerSegment = 8;

  const unsigned int kNumBristles = 100;
  const unsigned int kNumVerticesPerBristle = 10;
  const float kBristleLength = 4.5;
  const float kBristleJitter = 0.5;

  const unsigned int kNumIterations = 20;
  const float kGravity = 30.0;
  const float kBrushDamping = 0.75;
  const float kStiffnessVariation = 0.3;

  // Non-constant variables.
  float position_x_, position_y_, position_z_;
  float scale_;
  bool is_initialized_;

  std::deque<float> speeds_;

  typedef absl::flat_hash_map<std::string, GLuint> StrToLocMap;

  absl::flat_hash_map<ProgramId, GLuint> program_map_;
  absl::flat_hash_map<ProgramId, StrToLocMap> attrib_loc_map_;
  absl::flat_hash_map<ProgramId, StrToLocMap> uniform_loc_map_;

  absl::flat_hash_map<TextureId, GLuint> texture_map_;

  GLuint quad_vbo_ = 0;
  absl::flat_hash_map<VboId, GLuint> vbo_map_;

  GLuint simulation_framebuffer_ = 0;

  int num_splat_indices_ = 0;
  int num_brush_indices_ = 0;
};

}  // namespace fluid
}  // namespace spiral

#endif  // SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_BRUSH_H_
