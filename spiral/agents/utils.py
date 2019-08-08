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

"""Common utilities used by SPIRAL agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import six
import sonnet as snt
import tensorflow as tf
import tensorflow_hub as hub


nest = tf.contrib.framework.nest


AgentOutput = collections.namedtuple(
    "AgentOutput", ["action", "policy_logits", "baseline"])
AgentState = collections.namedtuple(
    "AgentState", ["lstm_state", "prev_action"])


class ResidualStack(snt.AbstractModule):
  """A stack of ResNet V2 blocks."""

  def __init__(self,
               num_hiddens,
               num_residual_layers,
               num_residual_hiddens,
               filter_size=3,
               initializers=None,
               data_format="NHWC",
               activation=tf.nn.relu,
               name="residual_stack"):
    """Instantiate a ResidualStack."""
    super(ResidualStack, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
    self._filter_size = filter_size
    self._initializers = initializers
    self._data_format = data_format
    self._activation = activation

  def _build(self, h):
    for i in range(self._num_residual_layers):
      h_i = self._activation(h)

      h_i = snt.Conv2D(
          output_channels=self._num_residual_hiddens,
          kernel_shape=(self._filter_size, self._filter_size),
          stride=(1, 1),
          initializers=self._initializers,
          data_format=self._data_format,
          name="res_nxn_%d" % i)(h_i)
      h_i = self._activation(h_i)

      h_i = snt.Conv2D(
          output_channels=self._num_hiddens,
          kernel_shape=(1, 1),
          stride=(1, 1),
          initializers=self._initializers,
          data_format=self._data_format,
          name="res_1x1_%d" % i)(h_i)
      h += h_i
    return self._activation(h)


class ConvEncoder(snt.AbstractModule):
  """Convolutional encoder."""

  def __init__(self,
               factor_h,
               factor_w,
               num_hiddens,
               num_residual_layers,
               num_residual_hiddens,
               initializers=None,
               data_format="NHWC",
               name="conv_encoder"):

    super(ConvEncoder, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
    self._initializers = initializers
    self._data_format = data_format

    # Note that implicitly the network uses conv strides of 2.
    # input height / output height == factor_h.
    self._num_steps_h = factor_h.bit_length() - 1
    # input width / output width == factor_w.
    self._num_steps_w = factor_w.bit_length() - 1
    num_steps = max(self._num_steps_h, self._num_steps_w)
    if factor_h & (factor_h - 1) != 0:
      raise ValueError("`factor_h` must be a power of 2. It is %d" % factor_h)
    if factor_w & (factor_w - 1) != 0:
      raise ValueError("`factor_w` must be a power of 2. It is %d" % factor_w)
    self._num_steps = num_steps

  def _build(self, x):
    h = x
    for i in range(self._num_steps):
      stride = (2 if i < self._num_steps_h else 1,
                2 if i < self._num_steps_w else 1)
      h = snt.Conv2D(
          output_channels=self._num_hiddens,
          kernel_shape=(4, 4),
          stride=stride,
          initializers=self._initializers,
          data_format=self._data_format,
          name="strided_{}".format(i))(h)
      h = tf.nn.relu(h)

    h = snt.Conv2D(
        output_channels=self._num_hiddens,
        kernel_shape=(3, 3),
        stride=(1, 1),
        initializers=self._initializers,
        data_format=self._data_format,
        name="pre_stack")(h)

    h = ResidualStack(  # pylint: disable=not-callable
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens,
        initializers=self._initializers,
        data_format=self._data_format,
        name="residual_stack")(h)
    return h


class ConvDecoder(snt.AbstractModule):
  """Convolutional decoder."""

  def __init__(self,
               factor_h,
               factor_w,
               num_hiddens,
               num_residual_layers,
               num_residual_hiddens,
               num_output_channels=3,
               initializers=None,
               data_format="NHWC",
               name="conv_decoder"):
    super(ConvDecoder, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
    self._num_output_channels = num_output_channels
    self._initializers = initializers
    self._data_format = data_format

    # input height / output height == factor_h.
    self._num_steps_h = factor_h.bit_length() - 1
    # input width / output width == factor_w.
    self._num_steps_w = factor_w.bit_length() - 1
    num_steps = max(self._num_steps_h, self._num_steps_w)
    if factor_h & (factor_h - 1) != 0:
      raise ValueError("`factor_h` must be a power of 2. It is %d" % factor_h)
    if factor_w & (factor_w - 1) != 0:
      raise ValueError("`factor_w` must be a power of 2. It is %d" % factor_w)
    self._num_steps = num_steps

  def _build(self, x):
    h = snt.Conv2D(
        output_channels=self._num_hiddens,
        kernel_shape=(3, 3),
        stride=(1, 1),
        initializers=self._initializers,
        data_format=self._data_format,
        name="pre_stack")(x)

    h = ResidualStack(  # pylint: disable=not-callable
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens,
        initializers=self._initializers,
        data_format=self._data_format,
        name="residual_stack")(h)

    for i in range(self._num_steps):
      # Does reverse striding -- puts stride-2s after stride-1s.
      stride = (2 if (self._num_steps - 1 - i) < self._num_steps_h else 1,
                2 if (self._num_steps - 1 - i) < self._num_steps_w else 1)
      h = snt.Conv2DTranspose(
          output_channels=self._num_hiddens,
          output_shape=None,
          kernel_shape=(4, 4),
          stride=stride,
          initializers=self._initializers,
          data_format=self._data_format,
          name="strided_transpose_{}".format(i))(h)
      h = tf.nn.relu(h)

    x_recon = snt.Conv2D(
        output_channels=self._num_output_channels,
        kernel_shape=(3, 3),
        stride=(1, 1),
        initializers=self._initializers,
        data_format=self._data_format,
        name="final")(h)

    return x_recon


def export_hub_module(agent_ctor,
                      observation_spec,
                      noise_dim,
                      module_path,
                      checkpoint_path,
                      name_transform_fn=None):
  """Exports the agent as a TF-Hub module.

  Args:
    agent_ctor: A function returning a Sonnet module for the agent.
    observation_spec: A nested dict of `Array` specs describing an observation
      coming from the environment.
    noise_dim: The dimensionality of the noise vector used by the agent.
    module_path: A path where to export the module to.
    checkpoint_path: A path where to load the weights for the module.
    name_transform_fn: An optional function to provide mapping between
      variable name in the module and the variable name in the checkpoint.
  """

  def module_fn():
    """Builds a graph for the TF-Hub module."""
    agent = agent_ctor()

    # Get the initial agent state tensor.
    initial_agent_state = agent.initial_state(1)

    # Create a bunch of placeholders for the step function inputs.
    step_type_ph = tf.placeholder(dtype=tf.int32, shape=(1,))
    observation_ph = nest.map_structure(
        lambda s: tf.placeholder(dtype=tf.dtypes.as_dtype(s.dtype),  # pylint: disable=g-long-lambda
                                 shape=(1,) + s.shape),
        observation_spec)
    observation_ph["noise_sample"] = tf.placeholder(
        dtype=tf.float32, shape=(1, noise_dim))
    agent_state_ph = nest.map_structure(
        lambda t: tf.placeholder(dtype=t.dtype, shape=t.shape),
        initial_agent_state)

    # Get the step function outputs.
    agent_output, agent_state = agent.step(
        step_type_ph, observation_ph, agent_state_ph)

    # Now we need to add the module signatures. TF Hub modules require inputs
    # to be flat dictionaries. Since the agent's methods accept multiple
    # argument some of which being nested dictionaries we gotta work
    # some magic in order flatten the structure of the placeholders.
    initial_state_output_dict = dict(
        state=initial_agent_state)
    initial_state_output_dict = dict(
        nest.flatten_with_joined_string_paths(initial_state_output_dict))
    step_inputs_dict = dict(
        step_type=step_type_ph,
        observation=observation_ph,
        state=agent_state_ph)
    step_inputs_dict = dict(
        nest.flatten_with_joined_string_paths(step_inputs_dict))
    step_outputs_dict = dict(
        action=agent_output.action,
        state=agent_state)
    step_outputs_dict = dict(
        nest.flatten_with_joined_string_paths(step_outputs_dict))

    hub.add_signature(
        "initial_state", outputs=initial_state_output_dict)
    hub.add_signature(
        "step", inputs=step_inputs_dict, outputs=step_outputs_dict)

  spec = hub.create_module_spec(module_fn, drop_collections=["sonnet"])
  spec.export(module_path,
              checkpoint_path=checkpoint_path,
              name_transform_fn=name_transform_fn)


def get_module_wrappers(module_path):
  """Returns python functions implementing the agent.

  Args:
    module_path: A path which should be used to load the agent from.

  Returns:
    A tuple of two functions:
      * A function that returns the initial state of the agent.
      * A function that performs a step.
  """
  g = tf.Graph()
  session = tf.Session(graph=g)

  with g.as_default():
    agent = hub.Module(module_path)

    def to_python_fn(session, signature):
      """Converts a symbolic function into a plain python functions."""
      inputs_ph = {
          k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in six.iteritems(agent.get_input_info_dict(signature))}
      outputs = agent(inputs=inputs_ph, signature=signature, as_dict=True)

      def fn(**kwargs):
        feed_dict = {inputs_ph[k]: kwargs[k] for k in six.iterkeys(inputs_ph)}
        return session.run(outputs, feed_dict=feed_dict)

      return fn

    raw_initial_state_fn = to_python_fn(session, "initial_state")
    raw_step_fn = to_python_fn(session, "step")
    init_op = tf.global_variables_initializer()

  g.finalize()
  session.run(init_op)

  def wrapped_step_fn(step_type, observation, prev_state):
    """A convenience wrapper for a raw step function."""
    step_type, observation = nest.map_structure(
        lambda t: np.expand_dims(t, 0),
        (step_type, observation))
    step_inputs_dict = dict(
        step_type=step_type,
        observation=observation)
    step_inputs_dict = dict(
        nest.flatten_with_joined_string_paths(step_inputs_dict))
    step_inputs_dict.update(prev_state)
    output = raw_step_fn(**step_inputs_dict)
    action = {k.replace("action/", ""): v
              for k, v in six.iteritems(output)
              if k.startswith("action/")}
    state = {k: v for k, v in six.iteritems(output) if k.startswith("state/")}
    action = nest.map_structure(lambda t: np.squeeze(t, 0), action)
    return action, state

  return raw_initial_state_fn, wrapped_step_fn
