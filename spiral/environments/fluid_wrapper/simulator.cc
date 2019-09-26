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

#include "spiral/environments/fluid_wrapper/simulator.h"

#include <algorithm>
#include <numeric>
#include <ostream>
#include <utility>

#include "spiral/environments/fluid_wrapper/brush.h"
#include "spiral/environments/fluid_wrapper/utils.h"
#include "absl/strings/string_view.h"

namespace spiral {
namespace fluid {

Simulator::Simulator(const Config::Simulator& config) :
    kNumJacobiIterations(config.num_jacobi_iterations()),
    kNumFramesToSimulate(config.num_frames_to_simulate()),
    kSplatPadding(config.splat_padding()),
    kSpeedPadding(config.speed_padding()) {}

void Simulator::Setup(int width, int height,
                      absl::string_view shader_base_dir) {
  width_ = width;
  height_ = height;

  // Create shader programs.
  ShaderSourceComposer composer(shader_base_dir);
  program_map_[ProgramId::kSplat] = CreateShaderProgram(
      composer.Compose("splat.vert"),
      composer.Compose("splat.frag"),
      &attrib_loc_map_[ProgramId::kSplat],
      &uniform_loc_map_[ProgramId::kSplat]);
  program_map_[ProgramId::kVelocitySplat] = CreateShaderProgram(
      composer.Compose("splat.vert",
                      "#define VELOCITY \n"),
      composer.Compose("splat.frag",
                      "#define VELOCITY \n"),
      &attrib_loc_map_[ProgramId::kVelocitySplat],
      &uniform_loc_map_[ProgramId::kVelocitySplat]);
  program_map_[ProgramId::kAdvect] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("advect.frag"),
      &attrib_loc_map_[ProgramId::kAdvect],
      &uniform_loc_map_[ProgramId::kAdvect]);
  program_map_[ProgramId::kDivergence] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("divergence.frag"),
      &attrib_loc_map_[ProgramId::kDivergence],
      &uniform_loc_map_[ProgramId::kDivergence]);
  program_map_[ProgramId::kJacobi] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("jacobi.frag"),
      &attrib_loc_map_[ProgramId::kJacobi],
      &uniform_loc_map_[ProgramId::kJacobi]);
  program_map_[ProgramId::kSubtract] = CreateShaderProgram(
      composer.Compose("fullscreen.vert"),
      composer.Compose("subtract.frag"),
      &attrib_loc_map_[ProgramId::kSubtract],
      &uniform_loc_map_[ProgramId::kSubtract]);

  // Create VBOs.
  CHECK_GL_ERROR(glGenBuffers(1, &quad_vbo_));
  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(kQuadVerts),
                              kQuadVerts,
                              GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr));

  // Create simulation framebuffer.
  CHECK_GL_ERROR(glGenFramebuffers(1, &simulation_framebuffer_));

  // Create textures.
  for (auto texture_id : {
      TextureId::kPaint, TextureId::kPaintTemp,
      TextureId::kVelocity, TextureId::kVelocityTemp}) {
    texture_map_[texture_id] = CreateTexture(GL_RGBA32F, GL_RGBA, GL_FLOAT,
        width_, height_,
        nullptr,
        GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE,
        GL_LINEAR, GL_LINEAR);
  }
  for (auto texture_id : {TextureId::kDivergence, TextureId::kPressure,
                          TextureId::kPressureTemp}) {
    texture_map_[texture_id] = CreateTexture(GL_RGBA32F, GL_RGBA, GL_FLOAT,
        width_, height_,
        nullptr,
        GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE,
        GL_NEAREST, GL_NEAREST);
  }

  Reset();
}

void Simulator::Reset() {
  ClearTextures({
      TextureId::kPaint, TextureId::kPaintTemp,
      TextureId::kVelocity, TextureId::kVelocityTemp,
      TextureId::kPressure, TextureId::kPressureTemp,
      TextureId::kDivergence
  });
  splat_areas_.clear();
  frame_number_ = 0;
}

void Simulator::ClearTextures(std::vector<TextureId> texture_ids) {
  CHECK_GL_ERROR(
      glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));
  CHECK_GL_ERROR(glClearColor(0.0, 0.0, 0.0, 1.0));

  for (auto texture_id : texture_ids) {
    GLint texture = texture_map_.at(texture_id);
    CHECK_GL_ERROR(glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0));
    CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT));
  }
}

void Simulator::RunAdvect(Rectangle<float> simulation_area,
                          TextureId velocity_texture,
                          TextureId data_texture,
                          TextureId target_texture,
                          float delta_time,
                          float dissipation) {
  GLuint program = program_map_[ProgramId::kAdvect];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kAdvect];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kAdvect];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glEnable(GL_SCISSOR_TEST));
  CHECK_GL_ERROR(glScissor(simulation_area.xmin(),
                           simulation_area.ymin(),
                           simulation_area.Width(),
                           simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_dissipation"), dissipation));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"), width_, height_));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_velocityTexture"), 0,
      texture_map_.at(velocity_texture));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_inputTexture"), 1,
      texture_map_.at(data_texture));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_min"),
      simulation_area.xmin(), simulation_area.ymin()));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_max"),
      simulation_area.xmax(), simulation_area.ymax()));


  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(target_texture), 0));

  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_deltaTime"), delta_time));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  CHECK_GL_ERROR(glDisable(GL_SCISSOR_TEST));
}

void Simulator::RunDivergence(Rectangle<float> simulation_area) {
  GLuint program = program_map_[ProgramId::kDivergence];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kDivergence];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kDivergence];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glEnable(GL_SCISSOR_TEST));
  CHECK_GL_ERROR(glScissor(simulation_area.xmin(),
                           simulation_area.ymin(),
                           simulation_area.Width(),
                           simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"), width_, height_));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_velocityTexture"), 0,
      texture_map_.at(TextureId::kVelocity));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kDivergence), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  CHECK_GL_ERROR(glDisable(GL_SCISSOR_TEST));
}

void Simulator::RunJacobi(Rectangle<float> simulation_area) {
  GLuint program = program_map_[ProgramId::kJacobi];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kJacobi];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kJacobi];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glEnable(GL_SCISSOR_TEST));
  CHECK_GL_ERROR(glScissor(simulation_area.xmin(),
                           simulation_area.ymin(),
                           simulation_area.Width(),
                           simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"), width_, height_));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_divergenceTexture"), 1,
      texture_map_.at(TextureId::kDivergence));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kPressure), 0));
  CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT));

  for (int iteration = 0; iteration < kNumJacobiIterations; ++iteration) {
    CHECK_GL_ERROR(glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        texture_map_.at(TextureId::kPressureTemp), 0));
    SetTextureUniformVariable(
        uniform_loc_map.at("u_pressureTexture"), 1,
        texture_map_.at(TextureId::kPressure));

    CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

    CHECK_GL_ERROR(glDisable(GL_SCISSOR_TEST));

    std::swap(texture_map_.at(TextureId::kPressure),
              texture_map_.at(TextureId::kPressureTemp));
  }
}

void Simulator::RunSubtract(Rectangle<float> simulation_area) {
  GLuint program = program_map_[ProgramId::kSubtract];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kSubtract];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kSubtract];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(simulation_area.xmin(),
                            simulation_area.ymin(),
                            simulation_area.Width(),
                            simulation_area.Height()));

  CHECK_GL_ERROR(glEnable(GL_SCISSOR_TEST));
  CHECK_GL_ERROR(glScissor(simulation_area.xmin(),
                           simulation_area.ymin(),
                           simulation_area.Width(),
                           simulation_area.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_resolution"), width_, height_));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_pressureTexture"), 0,
      texture_map_.at(TextureId::kPressure));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_velocityTexture"), 1,
      texture_map_.at(TextureId::kVelocity));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kVelocityTemp), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  CHECK_GL_ERROR(glDisable(GL_SCISSOR_TEST));

  std::swap(texture_map_.at(TextureId::kVelocity),
            texture_map_.at(TextureId::kVelocityTemp));
}

void Simulator::RunSplat(const Brush &brush,
                         Rectangle<float> simulation_area,
                         Rectangle<float> painting_rectangle,
                         float radius,
                         float color[4],
                         float z_threshold) {
  GLuint program = program_map_[ProgramId::kSplat];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kSplat];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kSplat];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(0, 0, width_, height_));

  CHECK_GL_ERROR(glEnable(GL_SCISSOR_TEST));
  // Restrict splatting to area that'll be simulated
  CHECK_GL_ERROR(glScissor(simulation_area.xmin(),
                           simulation_area.ymin(),
                           simulation_area.Width(),
                           simulation_area.Height()));

  CHECK_GL_ERROR(glEnable(GL_BLEND));
  CHECK_GL_ERROR(glBlendEquation(GL_FUNC_ADD));
  CHECK_GL_ERROR(glBlendFuncSeparate(
      GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_paintingDimensions"),
      painting_rectangle.Width(), painting_rectangle.Height()));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_paintingPosition"),
      painting_rectangle.xmin(), painting_rectangle.ymin()));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_splatRadius"), radius));
  CHECK_GL_ERROR(glUniform4f(
      uniform_loc_map.at("u_splatColor"),
      color[0], color[1], color[2], color[3]));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_zThreshold"), z_threshold));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_positionsTexture"), 0,
      brush.GetTexture(Brush::TextureId::kPositions));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_previousPositionsTexture"), 1,
      brush.GetTexture(Brush::TextureId::kPreviousPositions));

  CHECK_GL_ERROR(glBindBuffer(
      GL_ARRAY_BUFFER, brush.GetVbo(Brush::VboId::kSplatCoords)));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_splatCoordinates"), 4, GL_FLOAT, false, 0, nullptr));
  CHECK_GL_ERROR(glBindBuffer(
      GL_ELEMENT_ARRAY_BUFFER, brush.GetVbo(Brush::VboId::kSplatIndices)));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kPaint), 0));

  CHECK_GL_ERROR(glDrawElements(
      GL_TRIANGLES, brush.NumSplatIndicesToRender(),
      GL_UNSIGNED_SHORT, nullptr));

  CHECK_GL_ERROR(glDisable(GL_BLEND));
  CHECK_GL_ERROR(glDisable(GL_SCISSOR_TEST));
}

void Simulator::RunVelocitySplat(const Brush &brush,
                                 Rectangle<float> simulation_area,
                                 Rectangle<float> painting_rectangle,
                                 float radius,
                                 float z_threshold,
                                 float velocity_scale) {
  GLuint program = program_map_[ProgramId::kVelocitySplat];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kVelocitySplat];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kVelocitySplat];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, simulation_framebuffer_));

  CHECK_GL_ERROR(glViewport(0, 0, width_, height_));

  CHECK_GL_ERROR(glEnable(GL_SCISSOR_TEST));
  // Restrict splatting to area that'll be simulated
  CHECK_GL_ERROR(glScissor(simulation_area.xmin(),
                           simulation_area.ymin(),
                           simulation_area.Width(),
                           simulation_area.Height()));

  CHECK_GL_ERROR(glEnable(GL_BLEND));
  CHECK_GL_ERROR(glBlendEquation(GL_FUNC_ADD));
  CHECK_GL_ERROR(glBlendFuncSeparate(
      GL_ONE, GL_ONE, GL_ZERO, GL_ZERO));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_paintingDimensions"),
      painting_rectangle.Width(), painting_rectangle.Height()));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_paintingPosition"),
      painting_rectangle.xmin(), painting_rectangle.ymin()));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_splatRadius"), radius));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_zThreshold"), z_threshold));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_velocityScale"), velocity_scale));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_positionsTexture"), 0,
      brush.GetTexture(Brush::TextureId::kPositions));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_previousPositionsTexture"), 1,
      brush.GetTexture(Brush::TextureId::kPreviousPositions));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_velocitiesTexture"), 2,
      brush.GetTexture(Brush::TextureId::kVelocities));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_previousVelocitiesTexture"), 3,
      brush.GetTexture(Brush::TextureId::kPreviousVelocities));

  CHECK_GL_ERROR(glBindBuffer(
      GL_ARRAY_BUFFER, brush.GetVbo(Brush::VboId::kSplatCoords)));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_splatCoordinates"), 4, GL_FLOAT, false, 0, nullptr));
  CHECK_GL_ERROR(glBindBuffer(
      GL_ELEMENT_ARRAY_BUFFER, brush.GetVbo(Brush::VboId::kSplatIndices)));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kVelocity), 0));

  CHECK_GL_ERROR(glDrawElements(
      GL_TRIANGLES, brush.NumSplatIndicesToRender(),
      GL_UNSIGNED_SHORT, nullptr));

  CHECK_GL_ERROR(glDisable(GL_BLEND));
  CHECK_GL_ERROR(glDisable(GL_SCISSOR_TEST));
}

Rectangle<float> Simulator::GetSimulationArea() {
  if (splat_areas_.empty()) {
    return Rectangle<float>(0, 0, 0, 0);
  }

  // Now let's work out the total simulation area we need to simulate.
  auto it = splat_areas_.begin();
  auto simulation_area = (*it).first;

  ++it;
  for (; it != splat_areas_.end(); ++it) {
    simulation_area = simulation_area.Union((*it).first);
  }

  simulation_area.Round();
  simulation_area = simulation_area.Intersect(
      Rectangle<float>(0, 0, width_, height_));

  return simulation_area;
}

void Simulator::CleanupSplatAreas() {
  // Remove all of the splat areas we no longer need to simulate.
  auto it = splat_areas_.begin();

  while (it != splat_areas_.end()) {
    if (frame_number_ - (*it).second > kNumFramesToSimulate) {
      it = splat_areas_.erase(it);
    } else {
      ++it;
    }
  }
}

void Simulator::Splat(const Brush &brush,
                      Rectangle<float> painting_rectangle,
                      float z_threshold,
                      float splat_color[4],
                      float splat_radius,
                      float velocity_scale) {
  // The area we need to simulate for this set of splats.
  int brush_padding = std::ceil(brush.Scale() * kSplatPadding);
  brush_padding += std::ceil(brush.GetFilteredSpeed() * kSpeedPadding);

  // We start in canvas space.
  Rectangle<float> area(brush.x() - brush_padding,
                        brush.y() - brush_padding,
                        brush_padding * 2,
                        brush_padding * 2);

  // Transform area into simulation space.
  area.Translate(-painting_rectangle.xmin(), -painting_rectangle.ymin());
  area.Scale(width_ / painting_rectangle.Width(),
             height_ / painting_rectangle.Height());
  area.Round();
  area = area.Intersect(Rectangle<float>(0, 0, width_, height_));

  splat_areas_.push_back(std::make_pair(area, frame_number_));

  auto simulation_area = GetSimulationArea();
  RunSplat(brush, simulation_area, painting_rectangle, splat_radius,
           splat_color, z_threshold);

  RunVelocitySplat(brush, simulation_area, painting_rectangle,
                   splat_radius, z_threshold, velocity_scale);
}

Simulator::Status Simulator::Simulate() {
  const float delta_time = 1.0 / 60.0;
  const float fluidity = 0.8;

  if (splat_areas_.empty()) {
    return Status::kSkippedSimulation;
  }

  auto simulation_area = GetSimulationArea();

  RunDivergence(simulation_area);
  RunJacobi(simulation_area);

  RunAdvect(
      simulation_area,
      TextureId::kVelocity, TextureId::kPaint, TextureId::kPaintTemp,
      delta_time, 1.0);
  std::swap(texture_map_.at(TextureId::kPaint),
            texture_map_.at(TextureId::kPaintTemp));

  RunAdvect(
      simulation_area,
      TextureId::kVelocity, TextureId::kVelocity, TextureId::kVelocityTemp,
      delta_time, fluidity);
  std::swap(texture_map_.at(TextureId::kVelocity),
            texture_map_.at(TextureId::kVelocityTemp));

  frame_number_++;

  CleanupSplatAreas();

  // If we finished simulating on this step clear all velocity textures.
  if (splat_areas_.empty()) {
    ClearTextures({TextureId::kVelocity, TextureId::kVelocityTemp});
  }

  return Status::kFinishedSimulation;
}

}  // namespace fluid
}  // namespace spiral
