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

#ifndef SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_ERROR_H_
#define SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_ERROR_H_

#include <string>

namespace spiral {

// When an error is encountered, Spiral code should call SpielFatalError()
// which will forward the message to the current error handler.
// The default error handler outputs the error message to stderr, and exits
// the process with exit code 1.

// When called from Python, a different error handled is used, which returns
// RuntimeException to the caller, containing the error message.

// Report a runtime error.
[[noreturn]] void FatalError(const std::string& error_msg);

// Specify a new error handler.
using ErrorHandler = void (*)(const std::string&);
void SetErrorHandler(ErrorHandler error_handler);

}  // namespace spiral

#endif  // SPIRAL_ENVIRONMENTS_FLUID_WRAPPER_ERROR_H_
