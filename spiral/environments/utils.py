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

"""Utitlity functions for environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def quadratic_bezier(p_s, p_c, p_e, n):
  t = np.linspace(0., 1., n)
  t = t.reshape((1, n, 1))
  p_s, p_c, p_e = [np.expand_dims(p, axis=1) for p in [p_s, p_c, p_e]]
  p = (1 - t) * (1 - t) * p_s + 2 * (1 - t) * t * p_c + t * t * p_e
  return p


def rgb_to_hsv(red, green, blue):
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
