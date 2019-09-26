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

#include "spiral/environments/fluid_wrapper/error.h"

#include <iostream>

namespace spiral {

void DefaultErrorHandler(const std::string& error_msg) {
  std::cerr << "SPIRAL Fatal Error: " << error_msg << std::endl << std::endl;
  std::exit(1);
}

ErrorHandler error_handler = DefaultErrorHandler;

void SetErrorHandler(ErrorHandler new_error_handler) {
  error_handler = new_error_handler;
}

void FatalError(const std::string& error_msg) {
  error_handler(error_msg);
  // The error handler should not return. If it does, we will abort the process.
  std::cerr << "Error handler failure - exiting" << std::endl;
  std::exit(1);
}

}  // namespace spiral
