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

#ifndef SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_UTILS_H_
#define SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_UTILS_H_

#include <cmath>
#include <map>
#include <string>

#include "third_party/swiftshader/include/GLES3/gl3.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "spiral/environments/fluid_wrapper/error.h"

#define CHECK_GL_ERROR(glexpr)                                            \
  do {                                                                    \
    (glexpr);                                                             \
    GLenum error = glGetError();                                          \
    if (error != GL_NO_ERROR) {                                           \
      spiral::FatalError(                                                 \
          absl::StrCat("GL ERROR: 0x", absl::Hex(error, absl::kZeroPad4), \
                       " file:", __FILE__, ", line: ", __LINE__));        \
    }                                                                     \
  } while (false)

namespace spiral {
namespace fluid {

constexpr float kQuadVerts[] = {
    -1.0f, -1.0f,
    -1.0f, 1.0f,
    1.0f,  -1.0f,
    1.0f,  1.0f,
};

constexpr char kVertexShader[] = R"(
#version 330 core

layout(location = 0) in vec2 pos;
layout(location = 1) in vec3 color;

out vec3 vertex_color;

void main(){
  gl_Position = vec4(pos, 0.0, 1.0);
  vertex_color = color;
}
)";

constexpr char kFragmentShader[] = R"(
in vec3 vertex_color;

void main()
{
  gl_FragColor = vec4(vertex_color, 1.0);
}
)";

constexpr float kTriangleVerts[] = {
    -1.0f, 1.0f,   //
    0.0f,  -1.0f,  //
    1.0f,  1.0f,
};

constexpr float kTriangleVertColors[] = {
    1.0f, 0.0f, 0.0f,  //
    0.0f, 1.0f, 0.0f,  //
    0.0f, 0.0f, 1.0f,
};

template <typename T>
class Rectangle {
 public:
  static constexpr T kLargeNum = 9999;

  Rectangle() :
      xmin_(kLargeNum), ymin_(kLargeNum),
      xmax_(-kLargeNum), ymax_(-kLargeNum) {}
  Rectangle(const T& xmin, const T& ymin, const T& width, const T& height) :
      xmin_(xmin), ymin_(ymin), xmax_(xmin + width), ymax_(ymin + height) {}

  static Rectangle FromMinMax(const T& xmin, const T& ymin,
                              const T& xmax, const T& ymax) {
    return Rectangle(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  T xmin() const { return xmin_; }
  T xmax() const { return xmax_; }
  T ymin() const { return ymin_; }
  T ymax() const { return ymax_; }

  T Width() const { return xmax_ - xmin_; }
  T Height() const  { return ymax_ - ymin_; }

  void Set(const T& x, const T& y, const T& width, const T& height) {
    xmin_ = x;
    xmax_ = x + width;
    ymin_ = y;
    ymax_ = y + height;
  }

  Rectangle Union(const Rectangle& other) const {
    return Rectangle::FromMinMax(
        std::min(xmin_, other.xmin()), std::min(ymin_, other.ymin()),
        std::max(xmax_, other.xmax()), std::max(ymax_, other.ymax()));
  }

  Rectangle Intersect(const Rectangle& other) const {
    T xmin = std::max(xmin_, other.xmin());
    T ymin = std::max(ymin_, other.ymin());
    T xmax = std::min(xmax_, other.xmax());
    T ymax = std::min(ymax_, other.ymax());

    if (xmin > xmax || ymin > ymax)
      return Rectangle();
    else
      return Rectangle::FromMinMax(xmin, ymin, xmax, ymax);
  }

  void Translate(const T& dx, const T& dy) {
    xmin_ += dx;
    xmax_ += dx;
    ymin_ += dy;
    ymax_ += dy;
  }

  void Scale(float scale_x, float scale_y) {
    float xmid = 0.5 * (xmin_ + xmax_);
    float ymid = 0.5 * (ymin_ + ymax_);
    float width = Width() * scale_x;
    float height = Height() * scale_y;
    float xmin = xmid - 0.5 * width;
    float ymin = ymid - 0.5 * height;
    Set(xmin, ymin, width, height);
  }

  void Round() {
    xmin_ = std::round(xmin_);
    xmax_ = std::round(xmax_);
    ymin_ = std::round(ymin_);
    ymax_ = std::round(ymax_);
  }

 private:
  T xmin_;
  T xmax_;
  T ymin_;
  T ymax_;
};

class ShaderSourceComposer {
 public:
  explicit ShaderSourceComposer(absl::string_view base_dir) :
      base_dir_(base_dir) {}

  std::string Compose(absl::string_view shader_filename,
                      absl::string_view macro = absl::string_view());

 private:
  std::string base_dir_;
};

GLuint CompileShader(GLenum type, absl::string_view text);

GLuint CreateShaderProgram(
    absl::string_view vert_shader_text,
    absl::string_view frag_shader_text,
    absl::flat_hash_map<std::string, GLuint> *attrib_loc_map,
    absl::flat_hash_map<std::string, GLuint> *uniform_loc_map);

std::string ComposeShaderSource(
    absl::string_view shader_filename,
    absl::string_view macro = absl::string_view());

GLuint CreateTexture(GLint internal_format, GLint format,
                     GLenum type,
                     GLsizei width, GLsizei height,
                     const GLvoid *data,
                     GLint wrap_s, GLint wrap_t,
                     GLint min_filter, GLint mag_filter);

void SetTextureUniformVariable(
    GLuint uniform_loc, unsigned int unit, GLuint texture);

inline float mix(float a, float b, float t) {
  return (1.0 - t) * a + t * b;
}

// NOTE: I adapted it from the original code. Not sure why it messes up
//       RGB and RYB.
void HsvToRyb(float h, float s, float v, float *r, float *g, float *b);

}  // namespace fluid
}  // namespace spiral

#endif  // SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_UTILS_H_
