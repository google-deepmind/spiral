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

#include "spiral/environments/fluid_wrapper/utils.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace spiral {
namespace fluid {

std::string ShaderSourceComposer::Compose(
    absl::string_view shader_filename,
    absl::string_view macro) {
  std::stringstream str_stream;
  auto full_path = absl::StrCat(base_dir_, "/", shader_filename);
  str_stream << std::ifstream(full_path).rdbuf();
  std::string shader_text = str_stream.str();
  absl::string_view shader_text_view(shader_text);
  return absl::StrCat(macro, shader_text_view);
}

// Compiles a shader with the given type and returns the handle.
GLuint CompileShader(GLenum type, absl::string_view text) {
  auto shader_id = glCreateShader(type);
  if (shader_id != 0) {
    const char* text_str = text.data();
    const GLint length[1] = {static_cast<GLint>(text.size())};
    CHECK_GL_ERROR(glShaderSource(shader_id, 1, &text_str, length));
    CHECK_GL_ERROR(glCompileShader(shader_id));
    GLint shader_ok;
    CHECK_GL_ERROR(glGetShaderiv(shader_id, GL_COMPILE_STATUS, &shader_ok));
    if (!shader_ok) {
      GLsizei log_length;
      CHECK_GL_ERROR(
          glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length));

      std::vector<char> info_log(log_length + 1);
      CHECK_GL_ERROR(
          glGetShaderInfoLog(shader_id, log_length, nullptr, &info_log[0]));
      CHECK_GL_ERROR(glDeleteShader(shader_id));
      spiral::FatalError(
          absl::StrCat("Failed to compile shader with error: ", &info_log[0]));
    }
  }
  return shader_id;
}

// Compiles and links a shader program, using the provided vertex and fragment
// shaders.
GLuint CreateShaderProgram(
    absl::string_view vert_shader_text,
    absl::string_view frag_shader_text,
    absl::flat_hash_map<std::string, GLuint> *attrib_loc_map,
    absl::flat_hash_map<std::string, GLuint> *uniform_loc_map) {
  // Compile vertex shader
  auto vertex_shader = CompileShader(GL_VERTEX_SHADER, vert_shader_text);

  // Compile fragment shader
  auto fragment_shader = CompileShader(GL_FRAGMENT_SHADER, frag_shader_text);

  // Link program
  auto program_id = glCreateProgram();
  CHECK_GL_ERROR(glAttachShader(program_id, vertex_shader));
  CHECK_GL_ERROR(glAttachShader(program_id, fragment_shader));
  CHECK_GL_ERROR(glLinkProgram(program_id));

  CHECK_GL_ERROR(glDeleteShader(fragment_shader));
  CHECK_GL_ERROR(glDeleteShader(vertex_shader));

  GLint program_ok;
  CHECK_GL_ERROR(glGetProgramiv(program_id, GL_LINK_STATUS, &program_ok));

  if (!program_ok) {
    GLsizei log_length;
    CHECK_GL_ERROR(
        glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &log_length));

    std::vector<char> info_log(log_length + 1);
    CHECK_GL_ERROR(
        glGetProgramInfoLog(program_id, log_length, nullptr, &info_log[0]));
    CHECK_GL_ERROR(glDeleteProgram(program_id));
    spiral::FatalError(absl::StrCat(
        "Failed to compile shader program_id with error: ", &info_log[0]));
  }

  const GLuint buf_size = 100;
  GLint size;
  GLenum type;

  // Get attribute locations.
  GLint num_attrib;
  CHECK_GL_ERROR(glGetProgramiv(
      program_id, GL_ACTIVE_ATTRIBUTES, &num_attrib));
  for (int i = 0; i < num_attrib; ++i) {
    std::string name(buf_size, '\0');
    GLint location = -1;
    GLsizei name_len;
    CHECK_GL_ERROR(glGetActiveAttrib(
        program_id, i, buf_size, &name_len, &size, &type, &name[0]));
    name.resize(name_len);
    CHECK_GL_ERROR(location = glGetAttribLocation(program_id, &name[0]));
    (*attrib_loc_map)[name] = location;
  }

  // Get uniform locations.
  GLint num_uniform;
  CHECK_GL_ERROR(glGetProgramiv(
      program_id, GL_ACTIVE_UNIFORMS, &num_uniform));
  for (int i = 0; i < num_uniform; ++i) {
    std::string name(buf_size, '\0');
    GLint location = -1;
    GLsizei name_len = 0;
    CHECK_GL_ERROR(glGetActiveUniform(
        program_id, i, buf_size, &name_len, &size, &type, &name[0]));
    name.resize(name_len);
    CHECK_GL_ERROR(location = glGetUniformLocation(program_id, &name[0]));
    (*uniform_loc_map)[name] = location;
  }

  return program_id;
}

GLuint CreateTexture(GLint internal_format, GLint format,
                     GLenum type,
                     GLsizei width, GLsizei height,
                     const GLvoid *data,
                     GLint wrap_s, GLint wrap_t,
                     GLint min_filter, GLint mag_filter) {
  GLuint texture = 0;

  CHECK_GL_ERROR(glGenTextures(1, &texture));
  CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, texture));

  CHECK_GL_ERROR(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s));
  CHECK_GL_ERROR(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t));
  CHECK_GL_ERROR(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter));
  CHECK_GL_ERROR(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter));

  CHECK_GL_ERROR(
      glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                   format, type, data));

  return texture;
}

void SetTextureUniformVariable(
    GLuint uniform_loc, unsigned int unit, GLuint texture) {
  CHECK_GL_ERROR(glUniform1i(uniform_loc, unit));
  CHECK_GL_ERROR(glActiveTexture(GL_TEXTURE0 + unit));
  CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, texture));
}

void HsvToRyb(float h, float s, float v, float *r, float *g, float *b) {
  const float c = v * s;
  const float h_dash = h * 6;

  const float x = c * (1.0 - std::abs(std::fmod(h_dash, 2.0f) - 1.0));

  const int mod = std::floor(h_dash);

  const float rs[6] = {c, x, 0, 0, x, c};
  const float gs[6] = {x, c, c, x, 0, 0};
  const float bs[6] = {0, 0, x, c, c, x};

  *r = rs[mod],
  *g = gs[mod],
  *b = bs[mod];

  const float m = v - c;

  *r += m;
  *g += m;
  *b += m;
}

}  // namespace fluid
}  // namespace spiral
