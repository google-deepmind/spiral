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

#ifndef SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_EGL_H_
#define SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_EGL_H_

#include "third_party/swiftshader/include/EGL/egl.h"
#include "third_party/swiftshader/include/EGL/eglext.h"

// The includes above use Xlib which ruins Status. Here, we undo the damage.
#if defined (Status)
#undef Status
typedef int Status;
#endif

#endif  // SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_EGL_H_
