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

#include "spiral/environments/fluid_wrapper/brush.h"

#include <algorithm>
#include <cstdint>
#include <random>

#include "absl/strings/string_view.h"
#include "spiral/environments/fluid_wrapper/utils.h"

namespace spiral {
namespace fluid {

Brush::Brush(const Config::Brush& config) :
    kNumPreviousSpeeds(config.num_previous_speeds()),
    kNumSplatsPerSegment(config.num_splats_per_segment()),
    kNumBristles(config.num_bristles()),
    kNumVerticesPerBristle(config.num_vertices_per_bristle()),
    kBristleLength(config.bristle_length()),
    kBristleJitter(config.bristle_jitter()),
    kNumIterations(config.num_iterations()),
    kGravity(config.gravity()),
    kBrushDamping(config.brush_damping()),
    kStiffnessVariation(config.stiffness_variation()),
    is_initialized_(false) {}

void Brush::Setup(absl::string_view shader_base_dir) {
  // Create shader programs.
  ShaderSourceComposer composer(shader_base_dir);
  program_map_[ProgramId::kProject] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("project.frag"),
      &attrib_loc_map_[ProgramId::kProject],
      &uniform_loc_map_[ProgramId::kProject]);
  program_map_[ProgramId::kDistanceConstraint] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("distanceconstraint.frag"),
      &attrib_loc_map_[ProgramId::kDistanceConstraint],
      &uniform_loc_map_[ProgramId::kDistanceConstraint]);
  program_map_[ProgramId::kPlaneConstraint] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("planeconstraint.frag"),
      &attrib_loc_map_[ProgramId::kPlaneConstraint],
      &uniform_loc_map_[ProgramId::kPlaneConstraint]);
  program_map_[ProgramId::kBendingConstraint] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("bendingconstraint.frag"),
      &attrib_loc_map_[ProgramId::kBendingConstraint],
      &uniform_loc_map_[ProgramId::kBendingConstraint]);
  program_map_[ProgramId::kSetBristles] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("setbristles.frag"),
      &attrib_loc_map_[ProgramId::kSetBristles],
      &uniform_loc_map_[ProgramId::kSetBristles]);
  program_map_[ProgramId::kUpdateVelocity] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("updatevelocity.frag"),
      &attrib_loc_map_[ProgramId::kUpdateVelocity],
      &uniform_loc_map_[ProgramId::kUpdateVelocity]);

  // Create VBOs.
  CHECK_GL_ERROR(glGenBuffers(1, &quad_vbo_));
  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(kQuadVerts),
                              kQuadVerts,
                              GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr));

  // Create brush texture coordinates buffer.
  std::vector<float> brush_texture_coords;
  for (int bristle_idx = 0; bristle_idx < kNumBristles; ++bristle_idx) {
    for (int vert_idx = 0; vert_idx < kNumVerticesPerBristle; ++vert_idx) {
      float texture_x = (bristle_idx + 0.5) / kNumBristles;
      float texture_y = (vert_idx + 0.5) / kNumVerticesPerBristle;
      brush_texture_coords.push_back(texture_x);
      brush_texture_coords.push_back(texture_y);
    }
  }

  CHECK_GL_ERROR(glGenBuffers(1, &vbo_map_[VboId::kBrushTextureCoords]));
  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER,
                              vbo_map_[VboId::kBrushTextureCoords]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              brush_texture_coords.size() * sizeof(float),
                              brush_texture_coords.data(),
                              GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr));

  // Create splat coordinates and indices buffers.
  std::vector<float> splat_coords;
  std::vector<uint16_t> splat_indices;
  uint16_t splat_index = 0;
  for (int bristle_idx = 0; bristle_idx < kNumBristles; ++bristle_idx) {
    for (int vert_idx = 0; vert_idx < kNumVerticesPerBristle - 1; ++vert_idx) {
      // We create a quad for each bristle vertex.
      for (int i = 0; i < kNumSplatsPerSegment; ++i) {
        float t = (i + 0.5) / kNumSplatsPerSegment;

        float texture_x = (bristle_idx + 0.5) / kNumBristles;
        float texture_y = (vert_idx + 0.5 + t) / kNumVerticesPerBristle;

        // Bottom left.
        splat_coords.push_back(texture_x);
        splat_coords.push_back(texture_y);
        splat_coords.push_back(-1);
        splat_coords.push_back(-1);

        // Bottom right.
        splat_coords.push_back(texture_x);
        splat_coords.push_back(texture_y);
        splat_coords.push_back(1);
        splat_coords.push_back(-1);

        // Top right.
        splat_coords.push_back(texture_x);
        splat_coords.push_back(texture_y);
        splat_coords.push_back(1);
        splat_coords.push_back(1);

        // Top left.
        splat_coords.push_back(texture_x);
        splat_coords.push_back(texture_y);
        splat_coords.push_back(-1);
        splat_coords.push_back(1);

        splat_indices.push_back(splat_index + 0);
        splat_indices.push_back(splat_index + 1);
        splat_indices.push_back(splat_index + 2);

        splat_indices.push_back(splat_index + 2);
        splat_indices.push_back(splat_index + 3);
        splat_indices.push_back(splat_index + 0);

        splat_index += 4;
      }
    }
  }
  num_splat_indices_ = splat_indices.size();

  CHECK_GL_ERROR(glGenBuffers(1, &vbo_map_[VboId::kSplatCoords]));
  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo_map_[VboId::kSplatCoords]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              splat_coords.size() * sizeof(float),
                              splat_coords.data(),
                              GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr));

  CHECK_GL_ERROR(glGenBuffers(1, &vbo_map_[VboId::kSplatIndices]));
  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              vbo_map_[VboId::kSplatIndices]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              splat_indices.size() * sizeof(uint16_t),
                              splat_indices.data(),
                              GL_STATIC_DRAW));
  // Creature brush index buffer.
  std::vector<uint16_t> brush_indices;
  for (int bristle_idx = 0; bristle_idx < kNumBristles; ++bristle_idx) {
    for (int vert_idx = 0; vert_idx < kNumVerticesPerBristle - 1; ++vert_idx) {
      uint16_t left = bristle_idx * kNumVerticesPerBristle + vert_idx;
      uint16_t right = bristle_idx * kNumVerticesPerBristle + vert_idx + 1;

      brush_indices.push_back(left);
      brush_indices.push_back(right);
    }
  }
  num_brush_indices_ = brush_indices.size();

  CHECK_GL_ERROR(glGenBuffers(1, &vbo_map_[VboId::kBrushIndices]));
  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              vbo_map_[VboId::kBrushIndices]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              brush_indices.size() * sizeof(uint16_t),
                              brush_indices.data(),
                              GL_STATIC_DRAW));

  // Create simulation framebuffer.
  CHECK_GL_ERROR(glGenFramebuffers(1, &simulation_framebuffer_));

  // Create textures.
  for (auto texture_id : {
      TextureId::kPositions, TextureId::kPreviousPositions,
      TextureId::kVelocities, TextureId::kPreviousVelocities,
      TextureId::kProjectedPositions, TextureId::kProjectedPositionsTemp}) {
    texture_map_[texture_id] = CreateTexture(
        GL_RGBA32F, GL_RGBA, GL_FLOAT,
        kNumBristles, kNumVerticesPerBristle,
        nullptr,
        GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE,
        GL_LINEAR, GL_LINEAR);
  }

  static std::uniform_real_distribution<float> distribution(0.0, 1.0);
  static std::default_random_engine generator;

  std::vector<float> randoms(kNumBristles * kNumVerticesPerBristle * 4);
  std::generate(randoms.begin(), randoms.end(),
                []() { return distribution(generator); });
  texture_map_[TextureId::kRandoms] = CreateTexture(
      GL_RGBA32F, GL_RGBA, GL_FLOAT,
      kNumBristles, kNumVerticesPerBristle,
      randoms.data(),
      GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE,
      GL_LINEAR, GL_LINEAR);

  is_initialized_ = false;
}

void Brush::ClearTextures(std::vector<TextureId> texture_ids) {
  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));
  CHECK_GL_ERROR(glClearColor(0.0, 0.0, 0.0, 1.0));

  for (auto texture_id : texture_ids) {
    GLint texture = texture_map_.at(texture_id);
    CHECK_GL_ERROR(glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0));
    CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT));
  }
}

void Brush::RunSetBristles(Rectangle<GLint> simulation_area,
                           TextureId target_texture) {
  GLuint program = program_map_[ProgramId::kSetBristles];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kSetBristles];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kSetBristles];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform3f(
      uniform_loc_map.at("u_brushPosition"),
      position_x_, position_y_, position_z_));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_brushScale"), scale_));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_bristleCount"), kNumBristles));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_bristleLength"), kBristleLength));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_verticesPerBristle"), kNumVerticesPerBristle));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_jitter"), kBristleJitter));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"),
      kNumBristles, kNumVerticesPerBristle));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_randomsTexture"), 2,
      texture_map_.at(TextureId::kRandoms));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(target_texture), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
}

void Brush::RunProject(Rectangle<GLint> simulation_area) {
  GLuint program = program_map_[ProgramId::kProject];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kProject];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kProject];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_positionsTexture"), 0,
      texture_map_.at(TextureId::kPositions));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_velocitiesTexture"), 1,
      texture_map_.at(TextureId::kVelocities));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_gravity"), kGravity));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_damping"), kBrushDamping));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"),
      kNumBristles, kNumVerticesPerBristle));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kProjectedPositions), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
}

void Brush::RunDistanceConstraint(Rectangle<GLint> simulation_area, int pass) {
  GLuint program = program_map_[ProgramId::kDistanceConstraint];
  const auto& uniform_loc_map =
      uniform_loc_map_[ProgramId::kDistanceConstraint];
  const auto& attrib_loc_map =
      attrib_loc_map_[ProgramId::kDistanceConstraint];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_positionsTexture"), 0,
      texture_map_.at(TextureId::kProjectedPositions));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_pointCount"), kNumVerticesPerBristle));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_targetDistance"),
      scale_ * kBristleLength / (kNumVerticesPerBristle - 1)));
  CHECK_GL_ERROR(glUniform1i(
      uniform_loc_map.at("u_pass"), pass));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"),
      kNumBristles, kNumVerticesPerBristle));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kProjectedPositionsTemp), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  std::swap(texture_map_.at(TextureId::kProjectedPositions),
            texture_map_.at(TextureId::kProjectedPositionsTemp));
}

void Brush::RunBendingConstraint(Rectangle<GLint> simulation_area, int pass) {
  GLuint program = program_map_[ProgramId::kBendingConstraint];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kBendingConstraint];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kBendingConstraint];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_positionsTexture"), 0,
      texture_map_.at(TextureId::kProjectedPositions));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_randomsTexture"), 1,
      texture_map_.at(TextureId::kRandoms));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_pointCount"), kNumVerticesPerBristle));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_stiffnessVariation"), kStiffnessVariation));
  CHECK_GL_ERROR(glUniform1i(
      uniform_loc_map.at("u_pass"), pass));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"),
      kNumBristles, kNumVerticesPerBristle));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kProjectedPositionsTemp), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  std::swap(texture_map_.at(TextureId::kProjectedPositions),
            texture_map_.at(TextureId::kProjectedPositionsTemp));
}

void Brush::RunPlaneConstraint(Rectangle<GLint> simulation_area) {
  GLuint program = program_map_[ProgramId::kPlaneConstraint];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kPlaneConstraint];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kPlaneConstraint];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_positionsTexture"), 0,
      texture_map_.at(TextureId::kProjectedPositions));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"),
      kNumBristles, kNumVerticesPerBristle));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kProjectedPositionsTemp), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  std::swap(texture_map_.at(TextureId::kProjectedPositions),
            texture_map_.at(TextureId::kProjectedPositionsTemp));
}

void Brush::RunUpdateVelocity(Rectangle<GLint> simulation_area) {
  GLuint program = program_map_[ProgramId::kUpdateVelocity];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kUpdateVelocity];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kUpdateVelocity];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_positionsTexture"), 0,
      texture_map_.at(TextureId::kPositions));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_projectedPositionsTexture"), 1,
      texture_map_.at(TextureId::kProjectedPositions));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"),
      kNumBristles, kNumVerticesPerBristle));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kPreviousVelocities), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  std::swap(texture_map_.at(TextureId::kVelocities),
            texture_map_.at(TextureId::kPreviousVelocities));
}

void Brush::Reset() {
  is_initialized_ = false;
}

void Brush::Initialize(float x, float y, float z, float scale) {
  position_x_ = x;
  position_y_ = y;
  position_z_ = z;
  scale_ = scale;

  speeds_.clear();
  speeds_.push_back(0.0f);

  RunSetBristles(
      Rectangle<GLint>(0, 0, kNumBristles, kNumVerticesPerBristle),
      TextureId::kPositions);
  RunSetBristles(
      Rectangle<GLint>(0, 0, kNumBristles, kNumVerticesPerBristle),
      TextureId::kPreviousPositions);
  ClearTextures({
      TextureId::kVelocities, TextureId::kPreviousVelocities,
      TextureId::kProjectedPositions, TextureId::kProjectedPositionsTemp
  });

  is_initialized_ = true;
}

void Brush::Update(float x, float y, float z, float scale) {
  const float dx = x - position_x_;
  const float dy = y - position_y_;
  const float dz = z - position_z_;

  const float speed = std::sqrt(dx * dx + dy * dy + dz * dz);
  speeds_.push_back(speed);
  if (speeds_.size() > kNumPreviousSpeeds) {
    speeds_.pop_front();
  }

  position_x_ = x;
  position_y_ = y;
  position_z_ = z;
  scale_ = scale;

  Rectangle<GLint> default_simulation_area(
      0, 0, kNumBristles, kNumVerticesPerBristle);

  RunProject(default_simulation_area);

  for (int iteration = 0; iteration < kNumIterations; ++iteration) {
    // Set the base position of each bristle by setting
    // first vertex (first row).
    RunSetBristles(Rectangle<GLint>(0, 0, kNumBristles, 1),
                   TextureId::kProjectedPositions);

    for (int pass = 0; pass < 2; ++pass) {
      RunDistanceConstraint(default_simulation_area, pass);
    }

    for (int pass = 0; pass < 3; ++pass) {
      RunBendingConstraint(default_simulation_area, pass);
    }

    RunPlaneConstraint(default_simulation_area);
  }

  RunUpdateVelocity(default_simulation_area);

  std::swap(texture_map_.at(TextureId::kPositions),
            texture_map_.at(TextureId::kPreviousPositions));
  std::swap(texture_map_.at(TextureId::kPositions),
            texture_map_.at(TextureId::kProjectedPositions));
}

}  // namespace fluid
}  // namespace spiral
