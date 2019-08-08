# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LibMyPaint Reinforcement Learning environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-import-not-at-top

import collections
import copy
import os

import dm_env as environment
from dm_env import specs
import enum
import numpy as np
from six.moves import xrange
import tensorflow as tf

from spiral.environments import pylibmypaint


nest = tf.contrib.framework.nest


class BrushSettings(enum.IntEnum):
  """Enumeration of brush settings."""

  (MYPAINT_BRUSH_SETTING_OPAQUE,
   MYPAINT_BRUSH_SETTING_OPAQUE_MULTIPLY,
   MYPAINT_BRUSH_SETTING_OPAQUE_LINEARIZE,
   MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC,
   MYPAINT_BRUSH_SETTING_HARDNESS,
   MYPAINT_BRUSH_SETTING_ANTI_ALIASING,
   MYPAINT_BRUSH_SETTING_DABS_PER_BASIC_RADIUS,
   MYPAINT_BRUSH_SETTING_DABS_PER_ACTUAL_RADIUS,
   MYPAINT_BRUSH_SETTING_DABS_PER_SECOND,
   MYPAINT_BRUSH_SETTING_RADIUS_BY_RANDOM,
   MYPAINT_BRUSH_SETTING_SPEED1_SLOWNESS,
   MYPAINT_BRUSH_SETTING_SPEED2_SLOWNESS,
   MYPAINT_BRUSH_SETTING_SPEED1_GAMMA,
   MYPAINT_BRUSH_SETTING_SPEED2_GAMMA,
   MYPAINT_BRUSH_SETTING_OFFSET_BY_RANDOM,
   MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED,
   MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED_SLOWNESS,
   MYPAINT_BRUSH_SETTING_SLOW_TRACKING,
   MYPAINT_BRUSH_SETTING_SLOW_TRACKING_PER_DAB,
   MYPAINT_BRUSH_SETTING_TRACKING_NOISE,
   MYPAINT_BRUSH_SETTING_COLOR_H,
   MYPAINT_BRUSH_SETTING_COLOR_S,
   MYPAINT_BRUSH_SETTING_COLOR_V,
   MYPAINT_BRUSH_SETTING_RESTORE_COLOR,
   MYPAINT_BRUSH_SETTING_CHANGE_COLOR_H,
   MYPAINT_BRUSH_SETTING_CHANGE_COLOR_L,
   MYPAINT_BRUSH_SETTING_CHANGE_COLOR_HSL_S,
   MYPAINT_BRUSH_SETTING_CHANGE_COLOR_V,
   MYPAINT_BRUSH_SETTING_CHANGE_COLOR_HSV_S,
   MYPAINT_BRUSH_SETTING_SMUDGE,
   MYPAINT_BRUSH_SETTING_SMUDGE_LENGTH,
   MYPAINT_BRUSH_SETTING_SMUDGE_RADIUS_LOG,
   MYPAINT_BRUSH_SETTING_ERASER,
   MYPAINT_BRUSH_SETTING_STROKE_THRESHOLD,
   MYPAINT_BRUSH_SETTING_STROKE_DURATION_LOGARITHMIC,
   MYPAINT_BRUSH_SETTING_STROKE_HOLDTIME,
   MYPAINT_BRUSH_SETTING_CUSTOM_INPUT,
   MYPAINT_BRUSH_SETTING_CUSTOM_INPUT_SLOWNESS,
   MYPAINT_BRUSH_SETTING_ELLIPTICAL_DAB_RATIO,
   MYPAINT_BRUSH_SETTING_ELLIPTICAL_DAB_ANGLE,
   MYPAINT_BRUSH_SETTING_DIRECTION_FILTER,
   MYPAINT_BRUSH_SETTING_LOCK_ALPHA,
   MYPAINT_BRUSH_SETTING_COLORIZE,
   MYPAINT_BRUSH_SETTING_SNAP_TO_PIXEL,
   MYPAINT_BRUSH_SETTING_PRESSURE_GAIN_LOG,
   MYPAINT_BRUSH_SETTINGS_COUNT) = range(46)


def quadratic_bezier(p_s, p_c, p_e, n):
  t = np.linspace(0., 1., n)
  t = t.reshape((1, n, 1))
  p_s, p_c, p_e = [np.expand_dims(p, axis=1) for p in [p_s, p_c, p_e]]
  p = (1 - t) * (1 - t) * p_s + 2 * (1 - t) * t * p_c + t * t * p_e
  return p


def _fix15_to_rgba(buf):
  """Converts buffer from a 15-bit fixed-point representation into uint8 RGBA.

  Taken verbatim from the C code for libmypaint.

  Args:
    buf: 15-bit fixed-point buffer represented in `uint16`.

  Returns:
    A `uint8` buffer with RGBA channels.
  """
  rgb, alpha = np.split(buf, [3], axis=2)
  rgb = rgb.astype(np.uint32)
  mask = alpha[..., 0] == 0
  rgb[mask] = 0
  rgb[~mask] = ((rgb[~mask] << 15) + alpha[~mask] // 2) // alpha[~mask]
  rgba = np.concatenate((rgb, alpha), axis=2)
  rgba = (255 * rgba + (1 << 15) // 2) // (1 << 15)
  return rgba.astype(np.uint8)


def _rgb_to_hsv(red, green, blue):
  """Converts RGB to HSV."""
  hue = 0.0

  red = np.clip(red, 0.0, 1.0)
  green = np.clip(green, 0.0, 1.0)
  blue = np.clip(blue, 0.0, 1.0)

  max_value = np.max([red, green, blue])
  min_value = np.min([red, green, blue])

  value = max_value
  delta = max_value - min_value

  if delta > 0.0001:
    saturation = delta / max_value

    if red == max_value:
      hue = (green - blue) / delta
      if hue < 0.0:
        hue += 6.0
    elif green == max_value:
      hue = 2.0 + (blue - red) / delta
    elif blue == max_value:
      hue = 4.0 + (red - green) / delta

    hue /= 6.0
  else:
    saturation = 0.0
    hue = 0.0

  return hue, saturation, value


class LibMyPaint(environment.Environment):
  """A painting environment wrapping libmypaint."""

  ACTION_NAMES = ["control", "end", "flag", "pressure", "size",
                  "red", "green", "blue"]
  SPATIAL_ACTIONS = ["control", "end"]
  COLOR_ACTIONS = ["red", "green", "blue"]
  BRUSH_APPEARANCE_PARAMS = ["pressure", "log_size",
                             "hue", "saturation", "value"]

  ACTION_MASKS = {
      "paint": collections.OrderedDict([
          ("control", 1.0),
          ("end", 1.0),
          ("flag", 1.0),
          ("pressure", 1.0),
          ("size", 1.0),
          ("red", 1.0),
          ("green", 1.0),
          ("blue", 1.0)]),
      "move": collections.OrderedDict([
          ("control", 0.0),
          ("end", 1.0),
          ("flag", 1.0),
          ("pressure", 0.0),
          ("size", 0.0),
          ("red", 0.0),
          ("green", 0.0),
          ("blue", 0.0)]),
  }

  STROKES_PER_STEP = 50
  DTIME = 0.1

  P_VALUES = np.linspace(0.1, 1.0, 10)
  R_VALUES = np.linspace(0.0, 1.0, 20)
  G_VALUES = np.linspace(0.0, 1.0, 20)
  B_VALUES = np.linspace(0.0, 1.0, 20)

  def __init__(self,
               episode_length,
               canvas_width,
               grid_width,
               brush_type,
               brush_sizes,
               use_color,
               use_pressure=True,
               use_alpha=False,
               background="white",
               rewards=None,
               discount=1.,
               brushes_basedir=""):
    self._name = "libmypaint"

    if brush_sizes is None:
      brush_sizes = [1, 2, 3]

    self._canvas_width = canvas_width
    self._grid_width = grid_width
    self._grid_size = grid_width * grid_width
    self._use_color = use_color
    self._use_alpha = use_alpha
    if not self._use_color:
      self._output_channels = 1
    elif not self._use_alpha:
      self._output_channels = 3
    else:
      self._output_channels = 4
    self._use_pressure = use_pressure
    assert np.all(np.array(brush_sizes) > 0.)
    self._log_brush_sizes = [np.log(float(i)) for i in brush_sizes]
    self._rewards = rewards

    # Build action specification and action masks.
    self._action_spec = collections.OrderedDict([
        ("control", specs.DiscreteArray(self._grid_size)),
        ("end", specs.DiscreteArray(self._grid_size)),
        ("flag", specs.DiscreteArray(2)),
        ("pressure", specs.DiscreteArray(len(self.P_VALUES))),
        ("size", specs.DiscreteArray(len(self._log_brush_sizes))),
        ("red", specs.DiscreteArray(len(self.R_VALUES))),
        ("green", specs.DiscreteArray(len(self.G_VALUES))),
        ("blue", specs.DiscreteArray(len(self.B_VALUES)))])
    self._action_masks = copy.deepcopy(self.ACTION_MASKS)

    def remove_action_mask(name):
      for k in self._action_masks.keys():
        del self._action_masks[k][name]

    if not self._use_pressure:
      del self._action_spec["pressure"]
      remove_action_mask("pressure")

    if len(self._log_brush_sizes) > 1:
      self._use_size = True
    else:
      del self._action_spec["size"]
      remove_action_mask("size")
      self._use_size = False

    if not self._use_color:
      for k in self.COLOR_ACTIONS:
        del self._action_spec[k]
        remove_action_mask(k)

    # Setup the painting surface.
    if background == "white":
      background = pylibmypaint.SurfaceWrapper.Background.kWhite
    elif background == "transparent":
      background = pylibmypaint.SurfaceWrapper.Background.kBlack
    else:
      raise ValueError(
          "Invalid background type: {}".format(background))
    self._surface = pylibmypaint.SurfaceWrapper(
        self._canvas_width, self._canvas_width, background)

    # Setup the brush.
    self._brush = pylibmypaint.BrushWrapper()
    self._brush.SetSurface(self._surface)
    self._brush.LoadFromFile(
        os.path.join(brushes_basedir, "brushes/{}.myb".format(brush_type)))

    self._episode_step = 0
    self._episode_length = episode_length
    self._prev_step_type = None
    self._discount = discount

  @property
  def name(self):
    """Gets the name of the environment."""
    return self._name

  @property
  def grid_width(self):
    return self._grid_width

  def _get_canvas(self):
    buf = self._surface.BufferAsNumpy()
    buf = buf.transpose((0, 2, 1, 3, 4))
    buf = buf.reshape((self._canvas_width, self._canvas_width, 4))
    canvas = np.single(_fix15_to_rgba(buf)) / 255.0
    return canvas

  def observation(self):
    canvas = self._get_canvas()
    if not self._use_color:
      canvas = canvas[..., 0:1]
    elif not self._use_alpha:
      canvas = canvas[..., 0:3]

    episode_step = np.array(self._episode_step, dtype=np.int32)
    episode_length = np.array(self._episode_length, dtype=np.int32)

    return collections.OrderedDict([
        ("canvas", canvas),
        ("episode_step", episode_step),
        ("episode_length", episode_length),
        ("action_mask", self._action_mask)])

  def _update_libmypaint_brush(self, **kwargs):
    if "log_size" in kwargs:
      self._brush.SetBaseValue(
          BrushSettings.MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC,
          kwargs["log_size"])

    hsv_keys = ["hue", "saturation", "value"]
    if any(k in kwargs for k in hsv_keys):
      assert all(k in kwargs for k in hsv_keys)
      self._brush.SetBaseValue(
          BrushSettings.MYPAINT_BRUSH_SETTING_COLOR_H, kwargs["hue"])
      self._brush.SetBaseValue(
          BrushSettings.MYPAINT_BRUSH_SETTING_COLOR_S, kwargs["saturation"])
      self._brush.SetBaseValue(
          BrushSettings.MYPAINT_BRUSH_SETTING_COLOR_V, kwargs["value"])

  def _update_brush_params(self, **kwargs):
    rgb_keys = ["red", "green", "blue"]

    if any(k in kwargs for k in rgb_keys):
      assert all(k in kwargs for k in rgb_keys)
      red, green, blue = [kwargs[k] for k in rgb_keys]
      for k in rgb_keys:
        del kwargs[k]
      if self._use_color:
        hue, saturation, value = _rgb_to_hsv(red, green, blue)
        kwargs.update(dict(
            hue=hue, saturation=saturation, value=value))

    self._prev_brush_params = copy.copy(self._brush_params)
    self._brush_params.update(kwargs)

    if not self._prev_brush_params["is_painting"]:
      # If we were not painting before we should pretend that the appearence
      # of the brush didn't change.
      self._prev_brush_params.update({
          k: self._brush_params[k] for k in self.BRUSH_APPEARANCE_PARAMS})

    # Update the libmypaint brush object.
    self._update_libmypaint_brush(**kwargs)

  def _reset_brush_params(self):
    hue, saturation, value = _rgb_to_hsv(
        self.R_VALUES[0], self.G_VALUES[0], self.B_VALUES[0])
    pressure = 0.0 if self._use_pressure else 1.0
    self._brush_params = collections.OrderedDict([
        ("y", 0.0),
        ("x", 0.0),
        ("pressure", pressure),
        ("log_size", self._log_brush_sizes[0]),
        ("hue", hue),
        ("saturation", saturation),
        ("value", value),
        ("is_painting", False)])
    self._prev_brush_params = None

    # Reset the libmypaint brush object.
    self._move_to(0.0, 0.0, update_brush_params=False)
    self._update_libmypaint_brush(**self._brush_params)

  def _move_to(self, y, x, update_brush_params=True):
    self._update_brush_params(y=y, x=y, is_painting=False)
    self._brush.Reset()
    self._brush.NewStroke()
    self._brush.StrokeTo(x, y, 0.0, self.DTIME)

  def _bezier_to(self, y_c, x_c, y_e, x_e, pressure,
                 log_size, red, green, blue):
    self._update_brush_params(
        y=y_e, x=x_e, pressure=pressure, log_size=log_size,
        red=red, green=green, blue=blue, is_painting=True)

    y_s, x_s, pressure_s = [
        self._prev_brush_params[k] for k in ["y", "x", "pressure"]]
    pressure_e = pressure

    # Compute point along the Bezier curve.
    p_s = np.array([[y_s, x_s]])
    p_c = np.array([[y_c, x_c]])
    p_e = np.array([[y_e, x_e]])
    points = quadratic_bezier(p_s, p_c, p_e, self.STROKES_PER_STEP + 1)[0]

    # We need to perform this pseudo-stroke at the beginning of the curve
    # so that libmypaint handles the pressure correctly.
    if not self._prev_brush_params["is_painting"]:
      self._brush.StrokeTo(x_s, y_s, pressure_s, self.DTIME)

    for t in xrange(self.STROKES_PER_STEP):
      alpha = float(t + 1) / self.STROKES_PER_STEP
      pressure = pressure_s * (1. - alpha) + pressure_e * alpha
      self._brush.StrokeTo(
          points[t + 1][1], points[t + 1][0], pressure, self.DTIME)

  def _grid_to_real(self, location):
    return tuple(self._canvas_width * float(c) / self._grid_width
                 for c in location)

  def _process_action(self, action):
    flag = action["flag"]

    # Get pressure and size.
    if self._use_pressure:
      pressure = self.P_VALUES[action["pressure"]]
    else:
      pressure = 1.0
    if self._use_size:
      log_size = self._log_brush_sizes[action["size"]]
    else:
      log_size = self._log_brush_sizes[0]
    if self._use_color:
      red = self.R_VALUES[action["red"]]
      green = self.G_VALUES[action["green"]]
      blue = self.B_VALUES[action["blue"]]
    else:
      red, green, blue = None, None, None

    # Get locations. NOTE: the order of the coordinates is (y, x).
    locations = [
        np.unravel_index(action[k], (self._grid_width, self._grid_width))
        for k in self.SPATIAL_ACTIONS]

    # Convert grid coordinates into full resolution coordinates.
    locations = [
        self._grid_to_real(location) for location in locations]

    return locations, flag, pressure, log_size, red, green, blue

  def reset(self):
    self._surface.Clear()
    self._reset_brush_params()

    self.stats = {
        "total_strokes": 0,
        "total_disjoint": 0,
    }

    # TODO: Use an all-zero action mask instead of the "move" mask here.
    #              Unfortunately, the agents we have rely on this bug (they
    #              take the mask as an input at the next time step).
    #              self._action_mask = nest.map_structure(
    #                  lambda _: 0.0, self._action_masks["move"])
    self._action_mask = self._action_masks["move"]

    time_step = environment.restart(observation=self.observation())
    self._episode_step = 1
    self._prev_step_type = time_step.step_type
    return time_step

  def step(self, action):
    """Performs an environment step."""
    # If the environment has just been created or finished an episode
    # we should reset it (ignoring the action).
    if self._prev_step_type in {None, environment.StepType.LAST}:
      return self.reset()

    for k in action.keys():
      self._action_spec[k].validate(action[k])

    locations, flag, pressure, log_size, red, green, blue = (
        self._process_action(action))
    loc_control, loc_end = locations

    # Perform action.
    self._surface.BeginAtomic()

    if flag == 1:  # The agent produces a visible stroke.
      self._action_mask = self._action_masks["paint"]
      y_c, x_c = loc_control
      y_e, x_e = loc_end
      self._bezier_to(y_c, x_c, y_e, x_e, pressure, log_size, red, green, blue)

      # Update episode statistics.
      self.stats["total_strokes"] += 1
      if not self._prev_brush_params["is_painting"]:
        self.stats["total_disjoint"] += 1
    elif flag == 0:  # The agent moves to a new location.
      self._action_mask = self._action_masks["move"]
      y_e, x_e = loc_end
      self._move_to(y_e, x_e)
    else:
      raise ValueError("Invalid flag value")

    self._surface.EndAtomic()

    # Handle termination of the episode.
    reward = 0.0
    self._episode_step += 1
    if self._episode_step == self._episode_length:
      time_step = environment.termination(reward=reward,
                                          observation=self.observation())
    else:
      time_step = environment.transition(reward=reward,
                                         observation=self.observation(),
                                         discount=self._discount)

    self._prev_step_type = time_step.step_type
    return time_step

  def observation_spec(self):
    action_mask_spec = nest.map_structure(
        lambda _: specs.Array(shape=(), dtype=np.float32),
        self._action_masks["move"])
    canvas_shape = (self._canvas_width,
                    self._canvas_width,
                    self._output_channels)
    return collections.OrderedDict([
        ("canvas", specs.Array(shape=canvas_shape, dtype=np.float32)),
        ("episode_step", specs.Array(shape=(), dtype=np.int32)),
        ("episode_length", specs.Array(shape=(), dtype=np.int32)),
        ("action_mask", action_mask_spec)])

  def action_spec(self):
    return self._action_spec

  def close(self):
    self._brush = None
    self._surface = None
