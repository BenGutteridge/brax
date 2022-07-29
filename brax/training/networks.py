# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""Network definitions."""

from typing import Any, Callable, Sequence, Tuple

import dataclasses
from flax import linen
import jax
import jax.numpy as jnp

from brax.training.spectral_norm import SNDense

default_recurrent_memory_size = 20

@dataclasses.dataclass
class FeedForwardModel:
  init: Any
  apply: Any


class GRU_MLP(linen.Module):
  """Standard MLP module with GRU layer inplace of final fully-connected layer."""
  layer_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, input: jnp.ndarray, hidden: jnp.ndarray):
    """
    Penultimate layer is a GRU cell with fixed sized memory.
    Its output goes through a final FC layer to ensure correct sized NN output
    Its hidden state is passed to the next timestep (to be an input to the GRU cell)
    """
    output = input
    penultimate = len(self.layer_sizes) - 2
    for i, layer_size in enumerate(self.layer_sizes):
      output = linen.Dense(
          layer_size,
          name=f'fc_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(output)
      if i == penultimate:
        # output = linen.LayerNorm()(output) # normalising inputs so we don't saturate tanh in GRU cell
        hidden, output = linen.GRUCell(gate_fn=linen.relu, 
                                       activation_fn=linen.relu, 
                                       name='gru_layer')(hidden, output)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        output = self.activation(output)
    return hidden, output


class LSTM_MLP(linen.Module):
  """Standard MLP module with GRU layer inplace of final fully-connected layer."""
  layer_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, input: jnp.ndarray, carry: Tuple[jnp.ndarray, jnp.ndarray]):
    """
    Penultimate layer is a GRU cell with fixed sized memory.
    Its output goes through a final FC layer to ensure correct sized NN output
    Its hidden state is passed to the next timestep (to be an input to the GRU cell)
    """
    output = input
    cell, hidden = carry
    penultimate = len(self.layer_sizes) - 2
    for i, layer_size in enumerate(self.layer_sizes):
      output = linen.Dense(
          layer_size,
          name=f'fc_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(output)
      if i == penultimate:
        (cell, hidden), output = linen.LSTMCell(name='lstm_layer')((cell, hidden), output)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        output = self.activation(output)
    return hidden, output


class MLP(linen.Module):
  """MLP module."""
  layer_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden


class SNMLP(linen.Module):
  """MLP module with Spectral Normalization."""
  layer_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = SNDense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden


def make_model(layer_sizes: Sequence[int],
               obs_size: int,
               activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
               spectral_norm: bool = False,
               recurrent: bool = False,
               memory_size = default_recurrent_memory_size # for recurrent model
               ) -> FeedForwardModel:
  """Creates a model.

  Args:
    layer_sizes: layers
    obs_size: size of an observation
    activation: activation
    spectral_norm: whether to use a spectral normalization (default: False).
    recurrent: whether to use a recurrent layer for the policy (default: False).

  Returns:
    a model
  """
  dummy_obs = jnp.zeros((1, obs_size))
  assert not (spectral_norm and recurrent) # not supported
  if recurrent:
    if recurrent == 'lstm':
      print('LSTM not defined yet.')
      assert False
    else:
      print('Using recurrent policy.')
      dummy_hidden = jnp.zeros((1, memory_size))
      module = GRU_MLP(layer_sizes, activation=activation)
      model = FeedForwardModel(init=lambda rng: module.init(rng, dummy_obs, dummy_hidden),
                              apply=module.apply)
  elif spectral_norm:
    module = SNMLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardModel(
        init=lambda rng1, rng2: module.init(
            {'params': rng1, 'sing_vec': rng2}, dummy_obs),
        apply=module.apply)
  else:
    module = MLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardModel(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
  return model


def make_models(policy_params_size: int,
                obs_size: int,
                pol_num_hidden_layers=4,          
                pol_num_neurons_per_layer=32,
                val_num_hidden_layers=5,
                val_num_neurons_per_layer = 256,
                recurrent=False,
                ) -> Tuple[FeedForwardModel, FeedForwardModel]:
  """Creates models for policy and value functions.

  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation

  Returns:
    a model for policy and a model for value function
  """

  # policy_model = make_model([32, 32, 32, 32, policy_params_size], obs_size) # OG
  # value_model = make_model([256, 256, 256, 256, 256, 1], obs_size) # OG

  pol_layer_sizes = [pol_num_neurons_per_layer] * pol_num_hidden_layers + [policy_params_size]
  val_layer_sizes = [val_num_neurons_per_layer] * val_num_hidden_layers + [1]

  policy_model = make_model(pol_layer_sizes, obs_size, recurrent=recurrent)
  value_model = make_model(val_layer_sizes, obs_size, recurrent=recurrent) 

  return policy_model, value_model
