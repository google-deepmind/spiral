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

#include "spiral/environments/libmypaint_wrapper/surface.h"

#include <algorithm>
#include <cstdint>

#include "glog/logging.h"
#include "absl/memory/memory.h"
#include "libmypaint/mypaint-surface.h"

namespace spiral {
namespace libmypaint {
namespace {

extern "C" {

static void free_simple_tiledsurf(MyPaintSurface* surface) {
  auto* tiled_surface = reinterpret_cast<MyPaintTiledSurface*>(surface);
  mypaint_tiled_surface_destroy(tiled_surface);
  delete static_cast<Surface*>(tiled_surface);
}

static void tile_request_start(MyPaintTiledSurface *tiled_surface,
                               MyPaintTileRequest *request) {
  Surface* self = static_cast<Surface*>(tiled_surface);

  const int tx = request->tx;
  const int ty = request->ty;

  uint16_t* tile_pointer = nullptr;

  if (tx >= self->tiles_width || ty >= self->tiles_height || tx < 0 || ty < 0) {
    // Give it a tile which we will ignore writes to.
    tile_pointer = self->null_tile.get();
  } else {
    // Compute the offset for the tile into our linear memory buffer of tiles.
    size_t rowstride = self->tiles_width * self->surface_tile_size;
    size_t x_offset = tx * self->surface_tile_size;
    size_t tile_offset = (rowstride * ty) + x_offset;

    tile_pointer = self->tile_buffer.get() + tile_offset / sizeof(uint16_t);
  }

  request->buffer = tile_pointer;
}

static void tile_request_end(MyPaintTiledSurface* tiled_surface,
                             MyPaintTileRequest* request) {
  Surface* self = static_cast<Surface*>(tiled_surface);

  const int tx = request->tx;
  const int ty = request->ty;

  if (tx >= self->tiles_width || ty >= self->tiles_height || tx < 0 || ty < 0) {
    // Wipe any changes done to the null tile.
    std::fill_n(
        self->null_tile.get(),
        self->tile_n_elems,
        self->background_value);
  } else {
    // We hand out direct pointers to our buffer, so for the normal
    // case nothing needs to be done.
  }
}

}  // extern "C"

// This implementation is good enough for our purposes.
inline int CeilOfRatio(int a, int b) {
  return (a + b - 1) / b;
}

Surface* surface_new(int width, int height, const uint16_t& background_value) {
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);

  Surface* self = new Surface();
  mypaint_tiled_surface_init(self, tile_request_start, tile_request_end);

  const int tile_size_pixels = self->tile_size;

  // MyPaintSurface vfuncs.
  self->parent.destroy = free_simple_tiledsurf;

  const int tiles_width = CeilOfRatio(width, tile_size_pixels);
  const int tiles_height = CeilOfRatio(height, tile_size_pixels);
  const size_t tile_n_elems = tile_size_pixels * tile_size_pixels * 4;
  const size_t buffer_n_elems = tiles_width * tiles_height * tile_n_elems;

  CHECK_GE(tile_size_pixels * tiles_width, width);
  CHECK_GE(tile_size_pixels * tiles_height, height);
  CHECK_GE(buffer_n_elems, width * height * 4);

  self->background_value = background_value;

  uint16_t* buffer = new uint16_t[buffer_n_elems];
  CHECK(buffer) << "Unable to allocate enough memory: "
                << buffer_n_elems << " elements";
  std::fill_n(buffer, buffer_n_elems, background_value);

  self->tile_buffer = std::unique_ptr<uint16_t[]>(buffer);
  self->tile_n_elems = tile_n_elems;
  self->surface_tile_size = tile_n_elems * sizeof(uint16_t);
  self->null_tile = absl::make_unique<uint16_t[]>(tile_n_elems);
  self->tiles_width = tiles_width;
  self->tiles_height = tiles_height;
  self->width = width;
  self->height = height;

  std::fill_n(self->null_tile.get(), tile_n_elems, background_value);

  return self;
}

}  // namespace

SurfaceWrapper::SurfaceWrapper(int width, int height, Background color)
    : surface_(surface_new(width, height,
                           color == Background::kWhite ? (1 << 15) : 0)) {}

SurfaceWrapper::~SurfaceWrapper() {
  mypaint_surface_unref(&surface_->parent);
}

void SurfaceWrapper::BeginAtomic() {
  mypaint_surface_begin_atomic(&surface_->parent);
}

void SurfaceWrapper::EndAtomic() {
  MyPaintRectangle roi;
  mypaint_surface_end_atomic(&surface_->parent, &roi);
}

void SurfaceWrapper::Clear() {
  size_t buffer_n_elems = surface_->tiles_width * surface_->tiles_height *
                          surface_->tile_n_elems;
  std::fill_n(surface_->tile_buffer.get(), buffer_n_elems,
              surface_->background_value);
}

MyPaintSurface* SurfaceWrapper::GetInterface() {
  return &surface_->parent;
}

uint16_t* SurfaceWrapper::GetBuffer() {
  return surface_->tile_buffer.get();
}

std::vector<int> SurfaceWrapper::GetBufferDims() const {
  return {surface_->tiles_height,
          surface_->tiles_width,
          surface_->tile_size,
          surface_->tile_size,
          4};
}

}  // namespace libmypaint
}  // namespace spiral
