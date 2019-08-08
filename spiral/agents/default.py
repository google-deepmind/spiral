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

"""Default SPIRAL agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import dm_env as environment
import six
import sonnet as snt
import tensorflow as tf

from spiral.agents import utils


nest = tf.contrib.framework.nest


# Spatial action arguments need to be treated in a special way.
LOCATION_KEYS = ["end", "control"]


def _xy_grids(batch_size, height, width):
  x_grid = tf.linspace(-1., 1., width, name="linspace")
  x_grid = tf.reshape(x_grid, [1, 1, width, 1])
  x_grid = tf.tile(x_grid, [batch_size, height, 1, 1])
  y_grid = tf.linspace(-1., 1., height, name="linspace")
  y_grid = tf.reshape(y_grid, [1, height, 1, 1])
  y_grid = tf.tile(y_grid, [batch_size, 1, width, 1])
  return x_grid, y_grid


class AutoregressiveHeads(snt.AbstractModule):
  """A module for autoregressive action heads."""

  ORDERS = {
      "libmypaint": ["flag", "end", "control", "size", "pressure",
                     "red", "green", "blue"],
  }

  def __init__(self,
               z_dim,
               embed_dim,
               action_spec,
               decoder_params,
               order,
               grid_height,
               grid_width,
               name="autoregressive_heads"):
    super(AutoregressiveHeads, self).__init__(name=name)

    self._z_dim = z_dim
    self._action_spec = action_spec
    self._grid_height = grid_height
    self._grid_width = grid_width

    # Filter the order of actions according to the actual action specification.
    order = self.ORDERS[order]
    self._order = [k for k in order if k in action_spec]

    with self._enter_variable_scope():
      self._action_embeds = collections.OrderedDict(
          [(k, snt.Linear(output_size=embed_dim,
                          name=k + "_action_embed"))
           for k in six.iterkeys(action_spec)])

      self._action_heads = []
      for k, v in six.iteritems(action_spec):
        if k in LOCATION_KEYS:
          decoder = utils.ConvDecoder(  # pylint: disable=not-callable
              **decoder_params)
          action_head = snt.Sequential([
              snt.BatchReshape([4, 4, -1]),
              decoder,
              snt.BatchFlatten()], name=k + "_action_head")
        else:
          output_size = v.maximum - v.minimum + 1
          action_head = snt.Linear(
              output_size=output_size, name=k + "_action_head")
        self._action_heads.append((k, action_head))
      self._action_heads = collections.OrderedDict(self._action_heads)

      self._residual_mlps = {}
      for k, v in six.iteritems(self._action_spec):
        self._residual_mlps[k] = snt.nets.MLP(
            output_sizes=[16, 32, self._z_dim], name=k + "_residual_mlp")

  def _build(self, z):
    logits = {}
    action = {}
    for k in self._order:
      logits[k] = tf.check_numerics(
          self._action_heads[k](z), "Logits for {k} are not valid")
      a = tf.squeeze(tf.multinomial(logits[k], num_samples=1), -1)
      a = tf.cast(a, tf.int32, name=k + "_action")
      action[k] = a
      depth = self._action_spec[k].maximum - self._action_spec[k].minimum + 1
      # Asserts actions are valid.
      assert_op = tf.assert_less_equal(a, tf.constant(depth, dtype=a.dtype))
      with tf.control_dependencies([assert_op]):
        if k in LOCATION_KEYS:
          if depth != self._grid_height * self._grid_width:
            raise AssertionError(
                "Action space {depth} != grid_height * grid_width "
                "{self._grid_height}x{self._grid_width}.")
          w = self._grid_width
          h = self._grid_height
          y = -1.0 + 2.0 * tf.cast(a // w, tf.float32) / (h - 1)
          x = -1.0 + 2.0 * tf.cast(a % w, tf.float32) / (w - 1)
          a_vec = tf.stack([y, x], axis=1)
        else:
          a_vec = tf.one_hot(a, depth)
      a_embed = self._action_embeds[k](a_vec)
      residual = self._residual_mlps[k](tf.concat([z, a_embed], axis=1))
      z = tf.nn.relu(z + residual)

    action = collections.OrderedDict(
        [(k, action[k]) for k in six.iterkeys(self._action_spec)])
    logits = collections.OrderedDict(
        [(k, logits[k]) for k in six.iterkeys(self._action_spec)])

    return logits, action


class Agent(snt.AbstractModule):
  """A module for the default agent."""

  def __init__(
      self,
      action_spec,
      input_shape,
      grid_shape,
      action_order,
      name="default"):
    """Initialises the agent."""

    super(Agent, self).__init__(name=name)

    self._action_order = action_order
    self._action_spec = collections.OrderedDict(action_spec)

    self._z_dim = 256

    input_height, input_width = input_shape
    self._grid_height, self._grid_width = grid_shape
    enc_factor_h = input_height // 8  # Height of feature after encoding is 8
    enc_factor_w = input_width // 8  # Width of feature after encoding is 8
    dec_factor_h = self._grid_height // 4  # Height of feature after core is 4
    dec_factor_w = self._grid_width // 4  # Width of feature after core is 4

    self._encoder_params = {
        "factor_h": enc_factor_h,
        "factor_w": enc_factor_w,
        "num_hiddens": 32,
        "num_residual_layers": 8,
        "num_residual_hiddens": 32,
    }
    self._decoder_params = {
        "factor_h": dec_factor_h,
        "factor_w": dec_factor_w,
        "num_hiddens": 32,
        "num_residual_layers": 8,
        "num_residual_hiddens": 32,
        "num_output_channels": 1,
    }

    with self._enter_variable_scope():
      self._core = snt.LSTM(self._z_dim)

  def initial_state(self, batch_size):
    return utils.AgentState(
        lstm_state=self._core.initial_state(batch_size),
        prev_action=nest.map_structure(
            lambda spec: tf.zeros((batch_size,) + spec.shape, dtype=spec.dtype),
            self._action_spec))

  def _maybe_reset_core_state(self, core_state, should_reset):
    with tf.control_dependencies(None):
      if should_reset.shape.is_fully_defined():
        batch_size = should_reset.shape[0]
      else:
        batch_size = tf.shape(should_reset)[0]
      initial_core_state = self._core.initial_state(batch_size)
    # Use a reset state for the selected elements in the batch.
    state = nest.map_structure(
        lambda i, s: tf.where(should_reset, i, s),
        initial_core_state, core_state)
    return state

  def _compute_condition(self, action, mask):
    mask = tuple(mask[k] for k in self._action_spec.keys())
    conds = []

    action = action.values()
    for k, a, m in zip(self._action_spec.keys(), action, mask):
      depth = self._action_spec[k].maximum - self._action_spec[k].minimum + 1
      embed = snt.Linear(16)
      if k in LOCATION_KEYS:
        if depth != self._grid_height * self._grid_width:
          raise AssertionError(
              "Action space {depth} != grid_height * grid_width "
              "{self._grid_height}x{self._grid_width}.")
        w = self._grid_width
        h = self._grid_height
        y = -1.0 + 2.0 * tf.cast(a // w, tf.float32) / (h - 1)
        x = -1.0 + 2.0 * tf.cast(a % w, tf.float32) / (w - 1)
        a_vec = tf.concat([y, x], axis=1)
      else:
        a_vec = tf.one_hot(a, depth)[:, 0, :]
      cond = embed(a_vec) * m
      conds.append(cond)
    cond = tf.concat(conds, axis=1)
    cond = snt.nets.MLP([64, 32, 32])(cond)
    return cond

  @snt.reuse_variables
  def _torso(self,
             observation,
             prev_action,
             should_reset):
    batch_size, x_h, x_w, _ = observation["canvas"].get_shape().as_list()
    x_grid, y_grid = _xy_grids(batch_size, x_h, x_w)

    should_reset = tf.squeeze(should_reset, -1)
    prev_action = nest.map_structure(lambda pa: tf.where(  # pylint: disable=g-long-lambda
        should_reset, tf.zeros_like(pa), pa), prev_action)

    spatial_inputs = [observation["canvas"]]
    spatial_inputs += [x_grid, y_grid]
    data = tf.concat(spatial_inputs, axis=-1)

    with tf.variable_scope("torso"):
      h = snt.Conv2D(32, [5, 5])(data)

      # Compute conditioning vector based on the previously taken action.
      prev_action = nest.map_structure(
          lambda pa: tf.expand_dims(pa, -1), prev_action)
      cond = self._compute_condition(prev_action, observation["action_mask"])
      # Adjust the conditioning vector according to the noise sample
      # provided to the model. This is inspired by the original GAN framework.
      # NOTE: Unlike in normal GANs, this noise sample is not the only source
      #       of stochasticity. Stochastic actions contribute as well.
      assert observation["noise_sample"].shape.ndims == 2
      cond += snt.nets.MLP([64, 32, 32])(observation["noise_sample"])
      cond = tf.reshape(cond, [batch_size, 1, 1, -1])

      h += cond
      h = tf.nn.relu(h)

      encoder = utils.ConvEncoder(**self._encoder_params)

      h = snt.BatchFlatten()(encoder(h))
      h = snt.Linear(256)(tf.nn.relu(h))

      return h

  @snt.reuse_variables
  def _head(self, core_output):
    with tf.variable_scope("head"):
      head = AutoregressiveHeads(
          z_dim=self._z_dim,
          embed_dim=16,
          action_spec=self._action_spec,
          grid_height=self._grid_height,
          grid_width=self._grid_width,
          decoder_params=self._decoder_params,
          order=self._action_order)

      logits, actions = head(  # pylint: disable=not-callable
          core_output)
      baseline = tf.squeeze(snt.Linear(1)(core_output), -1)

      return utils.AgentOutput(actions, logits, baseline)

  def step(self,
           step_type,
           observation,
           prev_state):
    """Computes a single step of the agent."""
    with self._capture_variables():
      should_reset = tf.equal(step_type, environment.StepType.FIRST)

      torso_output = self._torso(
          observation,
          prev_state.prev_action,
          should_reset)

      lstm_state = self._maybe_reset_core_state(
          prev_state.lstm_state, should_reset)
      core_output, new_core_state = self._core(torso_output, lstm_state)

      agent_output = self._head(core_output)

    new_state = utils.AgentState(
        prev_action=agent_output.action,
        lstm_state=new_core_state)

    return agent_output, new_state

  def _build(self, *args):  # Unused.
    # pylint: disable=no-value-for-parameter
    return self.step(*args)
    # pylint: enable=no-value-for-parameter
