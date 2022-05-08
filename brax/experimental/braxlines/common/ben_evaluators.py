import functools
import os
from typing import Dict, Tuple, Any
from brax.experimental.braxlines.training import utils as training_utils
from brax.experimental.braxlines.training.env import wrap
from brax.io import html
import jax
from jax import numpy as jnp
from brax import jumpy as jp

def my_rollout_env(
    params, # : Dict[str, Dict[str, jnp.ndarray]],
    env_fn,
    inference_fn,
    batch_size: int = 0,
    seed: int = 0,
    reset_args: Tuple[Any] = (),
    step_args: Tuple[Any] = (),
    step_fn_name: str = 'step',
):
  """Visualize environment."""
  params_1, params_2 = params # extract from list
  rng = jax.random.PRNGKey(seed=seed)
  rng, reset_key = jax.random.split(rng)
  env = env_fn(batch_size=batch_size)
  inference_fn = inference_fn or functools.partial(
      training_utils.zero_fn, action_size=env.action_size)
  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(getattr(env, step_fn_name))
  jit_inference_fn = jax.jit(inference_fn)
  states = []
  state = jit_env_reset(reset_key, *reset_args)
  while not jnp.all(state.done):
    states.append(state)
    tmp_key, rng = jax.random.split(rng)
    act_1 = jit_inference_fn(params_1, state.obs, tmp_key)
    act_2 = jit_inference_fn(params_2, state.obs, tmp_key)
    act = jp.concatenate()[act_1[:len(act_1)//2], act_2[len(act_2)//2:]]
    state = jit_env_step(state, act, *step_args)
  states.append(state)
  return env, states


def my_visualize_env(batch_size: int = 0,
                  output_path: str = None,
                  output_name: str = 'video',
                  verbose: bool = False,
                  **kwargs):
  """Visualize env."""
  env, states = my_rollout_env(batch_size=batch_size, **kwargs)
  if verbose:
    print(f'Collected {max(1, batch_size)} trajs of T={len(states)}')

  if output_path:
    output_name = os.path.splitext(output_name)[0]
    if batch_size:

      for i in range(batch_size):
        filename = f'{output_path}/{output_name}_eps{i:02}.html'
        html.save_html(
            filename,
            env.sys, [
                jax.tree_map(functools.partial(jnp.take, indices=i), state.qp)
                for state in states
            ],
            make_dir=True)
        if verbose:
          print(f'Saved {filename}')
    else:
      filename = f'{output_path}/{output_name}.html'
      html.save_html(
          filename, env.sys, [state.qp for state in states], make_dir=True)
      if verbose:
        print(f'Saved {filename}')

  return env, states