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

"""Fluid Paint Reinforcement Learning environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-import-not-at-top

import collections
import copy

import dm_env as environment
from dm_env import specs
import numpy as np
from six.moves import xrange
import tensorflow as tf

from spiral.environments import utils
from spiral.environments import config_pb2
from spiral.environments import pyfluid


nest = tf.contrib.framework.nest


def mix(a, b, t):
  return a * (1.0 - t) + b * t


def circle_mix(a, b, t):
  """Interpolates between `a` and `b` assuming they lie on a circle."""
  case = np.argmin([np.abs(b - a), np.abs(b - a - 1), np.abs(b - a + 1)])
  if case == 0:
    result = np.float32(mix(a, b, t))
  elif case == 1:
    result = np.float32(mix(a, b - 1, t)) % np.float32(1.0)
  else:  # case == 2
    result = np.float32(mix(a, b + 1, t)) % np.float32(1.0)
  if result == 1.0:  # Somehow, result can be 1.0 at this point.
    return np.float32(0.0)  # We make sure that in this case we return 0.0.
  else:
    return result


class FluidPaint(environment.Environment):
  """A painting environment wrapping Fluid Paint."""

  ACTION_NAMES = ["control", "end", "flag", "speed", "size",
                  "red", "green", "blue", "alpha"]
  SPATIAL_ACTIONS = ["control", "end"]
  BRUSH_APPEARANCE_PARAMS = ["size", "hue", "saturation", "value", "alpha"]

  ACTION_MASKS = {
      "paint": collections.OrderedDict([
          ("control", 1.0),
          ("end", 1.0),
          ("flag", 1.0),
          ("speed", 1.0),
          ("size", 1.0),
          ("red", 1.0),
          ("green", 1.0),
          ("blue", 1.0),
          ("alpha", 1.0)]),
      "move": collections.OrderedDict([
          ("control", 0.0),
          ("end", 1.0),
          ("flag", 1.0),
          ("speed", 1.0),
          ("size", 0.0),
          ("red", 0.0),
          ("green", 0.0),
          ("blue", 0.0),
          ("alpha", 0.0)]),
  }

  STROKES_PER_STEP = 5 * np.arange(2, 11)

  R_VALUES = np.linspace(0.0, 1.0, 20)
  G_VALUES = np.linspace(0.0, 1.0, 20)
  B_VALUES = np.linspace(0.0, 1.0, 20)
  A_VALUES = np.linspace(0.0, 1.0, 10)

  def __init__(self,
               episode_length,
               canvas_width,
               grid_width,
               brush_sizes,
               rewards=None,
               discount=1.,
               shaders_basedir=""):
    self._name = "fluid_paint"

    if brush_sizes is None:
      self._brush_sizes = [10.0, 30.0, 50.0]
    else:
      self._brush_sizes = brush_sizes

    self._canvas_width = canvas_width
    self._grid_width = grid_width
    self._grid_size = grid_width * grid_width
    self._rewards = rewards

    # Build action specification and action masks.
    self._action_spec = collections.OrderedDict([
        ("control", specs.DiscreteArray(self._grid_size)),
        ("end", specs.DiscreteArray(self._grid_size)),
        ("flag", specs.DiscreteArray(2)),
        ("speed", specs.DiscreteArray(len(self.STROKES_PER_STEP))),
        ("size", specs.DiscreteArray(len(self._brush_sizes))),
        ("red", specs.DiscreteArray(len(self.R_VALUES))),
        ("green", specs.DiscreteArray(len(self.G_VALUES))),
        ("blue", specs.DiscreteArray(len(self.B_VALUES))),
        ("alpha", specs.DiscreteArray(len(self.A_VALUES)))])
    self._action_masks = copy.deepcopy(self.ACTION_MASKS)

    self._brush_params = None
    self._prev_reward = 0

    config = config_pb2.Config()

    self._wrapper = pyfluid.Wrapper(config.SerializeToString())
    self._wrapper.Setup(
        self._canvas_width, self._canvas_width, shaders_basedir)

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
    canvas = self._wrapper.CanvasAsNumpy()[..., :3]
    canvas = np.single(canvas) / 255.0
    return canvas

  def observation(self):
    canvas = self._get_canvas()

    episode_step = np.array(self._episode_step, dtype=np.int32)
    episode_length = np.array(self._episode_length, dtype=np.int32)

    return collections.OrderedDict([
        ("canvas", canvas),
        ("episode_step", episode_step),
        ("episode_length", episode_length),
        ("action_mask", self._action_mask)])

  def _update_brush_params(self, **kwargs):
    rgb_keys = ["red", "green", "blue"]

    if any(k in kwargs for k in rgb_keys):
      assert all(k in kwargs for k in rgb_keys)
      red, green, blue = [kwargs[k] for k in rgb_keys]
      for k in rgb_keys:
        del kwargs[k]
      hue, saturation, value = utils.rgb_to_hsv(red, green, blue)
      kwargs.update(dict(
          hue=hue, saturation=saturation, value=value))

    self._prev_brush_params = copy.copy(self._brush_params)
    self._brush_params.update(kwargs)

    if not self._prev_brush_params["is_painting"]:
      # If we were not painting before we should pretend that the appearence
      # of the brush didn't change.
      self._prev_brush_params.update({
          k: self._brush_params[k] for k in self.BRUSH_APPEARANCE_PARAMS})

  def _reset_brush_params(self):
    hue, saturation, value = utils.rgb_to_hsv(
        self.R_VALUES[0], self.G_VALUES[0], self.B_VALUES[0])
    self._brush_params = collections.OrderedDict([
        ("y", 0.0),
        ("x", 0.0),
        ("size", self._brush_sizes[0]),
        ("hue", hue),
        ("saturation", saturation),
        ("value", value),
        ("alpha", self.A_VALUES[0]),
        ("is_painting", False)])
    self._prev_brush_params = None

  def _move_to(self, y, x, num_strokes):
    self._update_brush_params(y=y, x=y, is_painting=False)

    y_s, x_s = [self._prev_brush_params[k] for k in ["y", "x"]]
    y_e, x_e = y, x

    for i in xrange(num_strokes):
      t = float(i + 1) / num_strokes
      x = mix(x_s, x_e, t)
      y = mix(y_s, y_e, t)
      self._wrapper.Update(x, y, self._brush_params["size"], False)

  def _bezier_to(self, y_c, x_c, y_e, x_e, num_strokes,
                 size, red, green, blue, alpha):
    self._update_brush_params(
        y=y_e, x=x_e, size=size, red=red, green=green, blue=blue, alpha=alpha,
        is_painting=True)

    y_s, x_s = [self._prev_brush_params[k] for k in ["y", "x"]]

    # Compute point along the Bezier curve.
    p_s = np.array([[y_s, x_s]])
    p_c = np.array([[y_c, x_c]])
    p_e = np.array([[y_e, x_e]])
    points = utils.quadratic_bezier(p_s, p_c, p_e, num_strokes + 1)[0]

    def mix_for_key(a, b, t, key):
      if key == "hue":
        return circle_mix(a, b, t)
      else:
        return mix(a, b, t)

    keys = self.BRUSH_APPEARANCE_PARAMS
    values_s = [self._prev_brush_params[k] for k in keys]
    values_e = [self._brush_params[k] for k in keys]

    for i in xrange(num_strokes):
      t = float(i + 1) / num_strokes

      # Interpolate brush appearance parameters.
      params = collections.OrderedDict([
          (k, mix_for_key(value_s, value_e, t, k))
          for k, value_s, value_e in zip(keys, values_s, values_e)])

      self._wrapper.SetBrushColor(
          params["hue"], params["saturation"], params["value"], params["alpha"])
      self._wrapper.Update(
          points[i + 1][1], points[i + 1][0], params["size"], True)

  def _grid_to_real(self, location):
    return tuple(self._canvas_width * float(c) / self._grid_width
                 for c in location)

  def _process_action(self, action):
    flag = action["flag"]

    num_strokes = self.STROKES_PER_STEP[action["speed"]]
    size = self._brush_sizes[action["size"]]
    red = self.R_VALUES[action["red"]]
    green = self.G_VALUES[action["green"]]
    blue = self.B_VALUES[action["blue"]]
    alpha = self.A_VALUES[action["alpha"]]

    # Get locations. NOTE: the order of the coordinates is (y, x).
    locations = [
        np.unravel_index(action[k], (self._grid_width, self._grid_width))
        for k in self.SPATIAL_ACTIONS]

    # Convert grid coordinates into full resolution coordinates.
    locations = [
        self._grid_to_real(location) for location in locations]

    return locations, flag, num_strokes, size, red, green, blue, alpha

  def reset(self):
    self._wrapper.Reset()
    self._reset_brush_params()

    # The first call of `Update()` after `Reset()` initializes the brush.
    # We don't need to simulate movement to the initial position.
    self._wrapper.Update(0.0, 0.0, self._brush_params["size"], False)

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

    locations, flag, num_strokes, size, red, green, blue, alpha = (
        self._process_action(action))
    loc_control, loc_end = locations

    # Perform action.
    if flag == 1:  # The agent produces a visible stroke.
      self._action_mask = self._action_masks["paint"]
      y_c, x_c = loc_control
      y_e, x_e = loc_end
      self._bezier_to(y_c, x_c, y_e, x_e, num_strokes, size,
                      red, green, blue, alpha)

      # Update episode statistics.
      self.stats["total_strokes"] += 1
      if not self._prev_brush_params["is_painting"]:
        self.stats["total_disjoint"] += 1
    elif flag == 0:  # The agent moves to a new location.
      self._action_mask = self._action_masks["move"]
      y_e, x_e = loc_end
      self._move_to(y_e, x_e, num_strokes)
    else:
      raise ValueError("Invalid flag value")

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
                    3)
    return collections.OrderedDict([
        ("canvas", specs.Array(shape=canvas_shape, dtype=np.float32)),
        ("episode_step", specs.Array(shape=(), dtype=np.int32)),
        ("episode_length", specs.Array(shape=(), dtype=np.int32)),
        ("action_mask", action_mask_spec)])

  def action_spec(self):
    return self._action_spec

  def close(self):
    self._wrapper = None
