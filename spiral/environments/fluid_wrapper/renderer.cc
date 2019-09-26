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

#include "spiral/environments/fluid_wrapper/renderer.h"

#include <math.h>

#include <vector>

#include "spiral/environments/fluid_wrapper/simulator.h"
#include "spiral/environments/fluid_wrapper/utils.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace spiral {
namespace fluid {

Renderer::Renderer(const Config& config) :
    kBrushHeight(config.brush_height()),
    kZThreshold(config.z_threshold()),
    kSplatVelocityScale(config.splat_velocity_scale()),
    kSplatRadius(config.splat_radius()),
    kMinAlpha(config.min_alpha()),
    kMaxAlpha(config.max_alpha()),
    kNormalScale(config.normal_scale()),
    kRoughness(config.roughness()),
    kF0(config.f0()),
    kSpecularScale(config.specular_scale()),
    kDiffuseScale(config.diffuse_scale()),
    kLightDirection{
        config.light_direction().x(),
        config.light_direction().y(),
        config.light_direction().z()
    },
    simulator_(config.simulator()),
    brush_(config.brush()) {}

void Renderer::Setup(int width, int height, absl::string_view shader_base_dir) {
  width_ = width;
  height_ = height;

  // Create shader programs.
  ShaderSourceComposer composer(shader_base_dir);
  program_map_[ProgramId::kPainting] = CreateShaderProgram(
      composer.Compose("painting.vert"),
      composer.Compose("painting.frag"),
      &attrib_loc_map_[ProgramId::kPainting],
      &uniform_loc_map_[ProgramId::kPainting]);

  // Create VAO.
  CHECK_GL_ERROR(glGenVertexArrays(1, &vao_));
  CHECK_GL_ERROR(glBindVertexArray(vao_));
  CHECK_GL_ERROR(glEnableVertexAttribArray(0));

  // Create VBOs.
  CHECK_GL_ERROR(glGenBuffers(1, &quad_vbo_));
  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(kQuadVerts),
                              kQuadVerts,
                              GL_STATIC_DRAW));
  CHECK_GL_ERROR(
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr));

  // Create framebuffer.
  CHECK_GL_ERROR(glGenFramebuffers(1, &framebuffer_));

  // Create textures.
  texture_map_[TextureId::kCanvas] = CreateTexture(
      GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
      width_, height_,
      nullptr,
      GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE,
      GL_LINEAR, GL_LINEAR);

  // Setup simulator.
  simulator_.Setup(width_, height_, shader_base_dir);

  // Setup brush.
  brush_.Setup(shader_base_dir);
  SetBrushColor(0.5, 1.0, 1.0, 0.8);

  canvas_.resize(width_ * height_ * 4);
  Reset();
}

void Renderer::Cleanup() {
  CHECK_GL_ERROR(glDisableVertexAttribArray(0));
  CHECK_GL_ERROR(glBindVertexArray(0));
}

void Renderer::ClearCanvas() {
  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kCanvas), 0));
  CHECK_GL_ERROR(glClearColor(0.0, 0.0, 0.0, 1.0));
  CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
}

void Renderer::Reset() {
  brush_.Reset();
  simulator_.Reset();

  canvas_updated_ = true;
}

void Renderer::RunPainting(Rectangle<float> painting_rectangle) {
  GLuint program = program_map_[ProgramId::kPainting];
  const auto& uniform_loc_map = uniform_loc_map_[ProgramId::kPainting];
  const auto& attrib_loc_map = attrib_loc_map_[ProgramId::kPainting];

  CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_));

  CHECK_GL_ERROR(glViewport(painting_rectangle.xmin(),
                            painting_rectangle.ymin(),
                            painting_rectangle.Width(),
                            painting_rectangle.Height()));

  CHECK_GL_ERROR(glUseProgram(program));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_normalScale"), kNormalScale / 1.0));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_roughness"), kRoughness));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_diffuseScale"), kDiffuseScale));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_specularScale"), kSpecularScale));
  CHECK_GL_ERROR(glUniform1f(
      uniform_loc_map.at("u_F0"), kF0));
  CHECK_GL_ERROR(glUniform3f(
      uniform_loc_map.at("u_lightDirection"),
      kLightDirection[0], kLightDirection[1], kLightDirection[2]));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_paintingPosition"),
      painting_rectangle.xmin(), painting_rectangle.ymin()));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_paintingSize"),
      painting_rectangle.Width(), painting_rectangle.Height()));
  CHECK_GL_ERROR(glUniform2f(
      uniform_loc_map.at("u_paintingResolution"),
      simulator_.width(), simulator_.height()));
  SetTextureUniformVariable(
      uniform_loc_map.at("u_paintTexture"), 0,
      simulator_.GetTexture(Simulator::TextureId::kPaint));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_));
  CHECK_GL_ERROR(glVertexAttribPointer(
      attrib_loc_map.at("a_position"), 2, GL_FLOAT, false, 0, nullptr));

  CHECK_GL_ERROR(glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
      texture_map_.at(TextureId::kCanvas), 0));

  CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
}

void Renderer::Update(float x, float y, float scale, bool is_painting) {
  Rectangle<float> painting_rectangle(0, 0, width_, height_);

  // Update brush.
  if (brush_.IsInitialized()) {
    brush_.Update(x, y, kBrushHeight * scale, scale);
  } else {
    brush_.Initialize(x, y, kBrushHeight * scale, scale);
    return;
  }

  // Splat into paint and velocity textures.
  if (is_painting) {
    float splat_radius = kSplatRadius * brush_.Scale();

    float splat_color[4];
    HsvToRyb(brush_color_[0], brush_color_[1], brush_color_[2],
             &splat_color[0], &splat_color[1], &splat_color[2]);

    float alpha_coeff = brush_color_[3];
    float alpha = mix(kMinAlpha, kMaxAlpha, alpha_coeff);
    splat_color[3] = alpha;

    float splat_velocity_scale =
        kSplatVelocityScale * alpha;

    // Splat paint.
    simulator_.Splat(
        brush_, painting_rectangle, kZThreshold * brush_.Scale(),
        splat_color, splat_radius, splat_velocity_scale);
  }

  Simulator::Status simulator_status = simulator_.Simulate();
  if (simulator_status == Simulator::Status::kFinishedSimulation) {
    canvas_updated_ = true;
  }
}

void Renderer::SetBrushColor(float h, float s, float v, float a) {
  brush_color_ = {h, s, v, a};
}

void Renderer::Render() {
  if (canvas_updated_) {
    ClearCanvas();
    RunPainting(Rectangle<float>(0, 0, width_, height_));
    CHECK_GL_ERROR(glReadPixels(0, 0, width_, height_, GL_RGBA,
                                GL_UNSIGNED_BYTE,
                                &canvas_[0]));
    canvas_updated_ = false;
  }
}

}  // namespace fluid
}  // namespace spiral
