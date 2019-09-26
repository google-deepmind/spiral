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

#ifndef SPIRAL_ENVIRONMENTS_LIBMYPAINT_WRAPPER_SURFACE_H_
#define SPIRAL_ENVIRONMENTS_LIBMYPAINT_WRAPPER_SURFACE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "libmypaint/mypaint-config.h"
#include "libmypaint/mypaint-glib-compat.h"
#include "libmypaint/mypaint-tiled-surface.h"

namespace spiral {
namespace libmypaint {

struct Surface : MyPaintTiledSurface {
  // Size (in bytes) of single tile.
  std::size_t surface_tile_size;
  // Size (in elements) of single tile.
  std::size_t tile_n_elems;
  // Stores tiles in a linear chunk of memory (16bpc RGBA).
  std::unique_ptr<std::uint16_t[]> tile_buffer;
  // Single tile that we hand out and ignore writes to.
  std::unique_ptr<std::uint16_t[]> null_tile;
  // Background color value.
  std::uint16_t background_value;
  // Width in tiles.
  int tiles_width;
  // Height in tiles.
  int tiles_height;
  // Width in pixels.
  int width;
  // Height in pixels
  int height;
};

class SurfaceWrapper {
 public:
  enum Background { kWhite, kBlack };

  SurfaceWrapper(int width, int height, Background color = kWhite);
  ~SurfaceWrapper();

  SurfaceWrapper(const SurfaceWrapper&) = delete;
  void operator=(const SurfaceWrapper&) = delete;

  void BeginAtomic();
  void EndAtomic();
  void Clear();

  MyPaintSurface* GetInterface();
  uint16_t* GetBuffer();
  std::vector<int> GetBufferDims() const;

 private:
  Surface* const surface_;
};

}  // namespace libmypaint
}  // namespace spiral

#endif  // SPIRAL_ENVIRONMENTS_LIBMYPAINT_WRAPPER_SURFACE_H_
