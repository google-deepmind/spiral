# SPIRAL

<p align="center">
  <img width="75%" src="media/default_wgangp_celebahq64_gen_19steps.gif">
</p>

## Overview

This repository contains agents and environments described in the ICML'18
paper ["Synthesizing Programs for Images using Reinforced Adversarial Learning"](http://proceedings.mlr.press/v80/ganin18a.html).
For the time being, we are providing two simulators:
one based on [`libmypaint`](https://github.com/mypaint/libmypaint) and one
based on [`Fluid Paint`](https://github.com/dli/paint) (**NOTE:** our
implementation is written in `C++` whereas the original is in `javascript`).
Additionally, we supply a [Sonnet](https://github.com/deepmind/sonnet) module
for the unconditional agent as well as
[pre-trained model snapshots](https://tfhub.dev/s?q=spiral)
(9 agents from a single population for `libmypaint` and 1 agent for
`Fluid Paint`) available from [TF-Hub](https://www.tensorflow.org/hub).

If you feel an immediate urge to dive into the code the most relevant files are:

| Path | Description |
| :--- | :--- |
| [`spiral/agents/default.py`](spiral/agents/default.py) | The architecture of the agent |
| [`spiral/environments/libmypaint.py`](spiral/environments/libmypaint.py) | The `libmypaint`-based environment |
| [`spiral/environments/fluid.py`](spiral/environments/fluid.py) | The `Fluid Paint`-based environment |

## Reference

If this repository is helpful for your research please cite the following
publication:

```
@inproceedings{ganin2018synthesizing,
  title={Synthesizing Programs for Images using Reinforced Adversarial Learning},
  author={Ganin, Yaroslav and Kulkarni, Tejas and Babuschkin, Igor and Eslami, SM Ali and Vinyals, Oriol},
  booktitle={ICML},
  year={2018}
}
```

## Installation

Clone this repository and fetch the external submodules:

```shell
git clone https://github.com/deepmind/spiral.git
cd spiral
git submodule update --init --recursive
```

Install required packages:

```shell
apt-get install cmake pkg-config protobuf-compiler libjson-c-dev intltool libpython3-dev python3-pip
pip3 install six setuptools numpy scipy tensorflow==1.14 tensorflow-hub dm-sonnet==1.35
```

**WARNING:** Make sure that you have `cmake` **3.14** or later since we rely
on its capability to find `numpy` libraries. If your package manager doesn't
provide it follow the installation instructions from
[here](https://cmake.org/install/). You can check the version by
running `cmake --version `.

Finally, run the following command to install the SPIRAL package itself:

```shell
python3 setup.py develop --user
```

You will also need to obtain the brush files for the `libmypaint` environment
to work properly. These can be found
[here](https://github.com/mypaint/mypaint-brushes). For example, you can
place them in `third_party` folder like this:

```shell
wget -c https://github.com/mypaint/mypaint-brushes/archive/v1.3.0.tar.gz -O - | tar -xz -C third_party
```

Finally, the `Fluid Paint` environment depends on the shaders from the original
`javascript` [implementation](https://github.com/dli/paint). You can obtain
them by running the following commands:

```shell
git clone https://github.com/dli/paint third_party/paint
patch third_party/paint/shaders/setbristles.frag third_party/paint-setbristles.patch
```

Optionally, in order to be able to try out the package in the provided
`jupyter` [notebook](notebooks/spiral-demo.ipynb), you’ll need to install
the following packages:

```shell
pip3 install matplotlib jupyter
```


## Usage

For a basic example of how to use the package please follow
[this notebook](notebooks/spiral-demo.ipynb).

### Sampling from a pre-trained model

We provide [pre-trained models](https://tfhub.dev/s?q=spiral) for unconditional
19-step generation of [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)
images. Here is an example of how you can sample from an agent interacting
with the `libmypaint` environment:

```python
import matplotlib.pyplot as plt

import spiral.agents.default as default_agent
import spiral.agents.utils as agent_utils
import spiral.environments.libmypaint as libmypaint


# The path to a TF-Hub module.
MODULE_PATH = "https://tfhub.dev/deepmind/spiral/default-wgangp-celebahq64-gen-19steps/agent4/1"
# The folder containing `libmypaint` brushes.
BRUSHES_PATH = "the/path/to/libmypaint-brushes"

# Here, we create an environment.
env = libmypaint.LibMyPaint(episode_length=20,
                            canvas_width=64,
                            grid_width=32,
                            brush_type="classic/dry_brush",
                            brush_sizes=[1, 2, 4, 8, 12, 24],
                            use_color=True,
                            use_pressure=True,
                            use_alpha=False,
                            background="white",
                            brushes_basedir=BRUSHES_PATH)


# Now we load the agent from a snapshot.
initial_state, step = agent_utils.get_module_wrappers(MODULE_PATH)

# Everything is ready for sampling.
state = initial_state()
noise_sample = np.random.normal(size=(10,)).astype(np.float32)

time_step = env.reset()
for t in range(19):
    time_step.observation["noise_sample"] = noise_sample
    action, state = step(time_step.step_type, time_step.observation, state)
    time_step = env.step(action)

# Show the sample.
plt.close("all")
plt.imshow(time_step.observation["canvas"], interpolation="nearest")
```

### Converting a trained agent into a TF-Hub module

```python
import spiral.agents.default as default_agent
import spiral.agents.utils as agent_utils
import spiral.environments.libmypaint as libmypaint


# This where we're going to put our TF-Hub module.
TARGET_PATH = ...
# A path to a checkpoint of the trained model.
CHECKPOINT_PATH = ...

# We will need to create an environment in order to obtain the specifications
# for the agent's action and the observation.
env = libmypaint.LibMyPaint(...)

# Here, we wrap a Sonnet module constructor for our agent in a function.
# This is to avoid contaminating the default tensorflow graph.
def agent_ctor():
  return default_agent.Agent(action_spec=env.action_spec(),
                             input_shape=(64, 64),
                             grid_shape=(32, 32),
                             action_order="libmypaint")

# Finally, export a TF-Hub module. We need to specify which checkpoint to use
# to extract the weights for the agent. Since the variable names in the
# checkpoint may differ from the names in the Sonnet module produced by
# `agent_ctor`, we may also want to provide an appropriate name mapping
# function.
agent_utils.export_hub_module(agent_ctor=agent_ctor,
                              observation_spec=env.observation_spec(),
                              noise_dim=10,
                              module_path=TARGET_PATH,
                              checkpoint_path=CHECKPOINT_PATH,
                              name_transform_fn=lambda name: ...)
```

## Disclaimer

This is not an official Google product.
