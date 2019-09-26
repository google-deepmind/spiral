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

#include "spiral/environments/fluid_wrapper/wrapper.h"

#include <cstdio>
#include <memory>
#include <ostream>

#include "absl/strings/str_cat.h"
#include "spiral/environments/fluid_wrapper/error.h"
#include "config.pb.h"
#include "spiral/environments/fluid_wrapper/renderer.h"

// Helper macro to check for EGL errors.
#define CHECK_EGL_ERROR(egl_expr)                                          \
  do {                                                                     \
    (egl_expr);                                                            \
    auto error = eglGetError();                                            \
    if (error != EGL_SUCCESS) {                                            \
      return spiral::FatalError(                                           \
          absl::StrCat("EGL ERROR: 0x", absl::Hex(error, absl::kZeroPad4), \
                       " file:", __FILE__, ", line: ", __LINE__));         \
    }                                                                      \
  } while (false)

// clang-format off
// EGL Config settings, used to define what components are required, the desired
// size in bits and various other settings.
constexpr EGLint kConfigAttribs[] = {
  EGL_RED_SIZE, 8,
  EGL_GREEN_SIZE, 8,
  EGL_BLUE_SIZE, 8,
  EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
  EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
  EGL_DEPTH_SIZE, 0,
  EGL_STENCIL_SIZE, 0,
  EGL_NONE
};

// Attributes required to generate a correctly sized pixel buffer.
constexpr EGLint kPixelBufferAttribs[] = {
  EGL_WIDTH, 1,
  EGL_HEIGHT, 1,
  EGL_NONE,
};

constexpr EGLint const kContextAttrList[] = {
  EGL_CONTEXT_CLIENT_VERSION, 3,
  EGL_NONE,
};
// clang-format on

extern "C" EGLDisplay eglGetPlatformDisplayEXT(EGLenum platform,
                                               void* native_display,
                                               const EGLint* attrib_list);

namespace spiral {
namespace fluid {

Wrapper::Wrapper(const Config& config) : renderer_(config), is_setup_(false) {}

void Wrapper::Setup(int width, int height, const std::string& shader_base_dir) {
  width_ = width;
  height_ = height;

  CHECK_EGL_ERROR(egl_display_ = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_MESA,
                                                          nullptr, nullptr));

  EGLint major, minor;
  EGLBoolean success;
  CHECK_EGL_ERROR(success = eglInitialize(egl_display_, &major, &minor));
  if (!success) {
    spiral::FatalError("eglInitialize failed.");
  }

  // Choose EGL config.
  EGLint num_configs;
  EGLConfig egl_config;
  CHECK_EGL_ERROR(success = eglChooseConfig(egl_display_, kConfigAttribs,
                                            &egl_config, 1, &num_configs));
  if (!success) {
    return spiral::FatalError("Failed to choose a valid EGLConfig.");
  }

  // Create EGL surface.
  CHECK_EGL_ERROR(egl_surface_ = eglCreatePbufferSurface(
                      egl_display_, egl_config, kPixelBufferAttribs));

  // Create EGL context & make current.
  CHECK_EGL_ERROR(egl_context_ = eglCreateContext(egl_display_, egl_config,
                                                  EGL_NO_CONTEXT,
                                                  kContextAttrList));

  CHECK_EGL_ERROR(
      eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_));

  renderer_.Setup(width, height, shader_base_dir);

  is_setup_ = true;
}

Wrapper::~Wrapper() {
  if (!is_setup_) {
    return;
  }
  renderer_.Cleanup();
  eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroyContext(egl_display_, egl_context_);
  eglDestroySurface(egl_display_, egl_surface_);
  eglTerminate(egl_display_);
}

uint8_t* Wrapper::GetCanvas() {
  renderer_.Render();
  return renderer_.GetCanvas();
}

std::vector<int> Wrapper::GetCanvasDims() const {
  return {height_, width_, 4};
}

}  // namespace fluid
}  // namespace spiral
