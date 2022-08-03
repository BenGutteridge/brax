import brax
import jax
import sys
import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt
import brax.jumpy as jp
import os
from os.path import join
from brax.io import model, html
from collections import OrderedDict as odict
from brax.training.networks import default_recurrent_memory_size as recurrent_memory_size


def make_group_action_shapes(players, actions_per_player):
  group_action_shapes, count = {}, 0
  for i in players:
    group_action_shapes[i] = dict(
        indices=tuple(np.arange(count*actions_per_player, (count+1)*actions_per_player)),
        shape=(actions_per_player,),
        size=actions_per_player,
        )
    count += 1
  return odict(group_action_shapes)

def make_config(n_players=2, 
                walls=False, 
                output_path=False, # saves as a string to 'config.py'
                frozen_players=False,
                friction=0.,
                player_radius=3.,
                ball_init='near_p1',
                ):
  body_idx, n = {}, 0
  pitm = brax.Config(dt=0.10, substeps=20, dynamics_mode='pbd')

  # make ball
  ball = pitm.bodies.add(name='ball', mass=1)
  body_idx['ball'] = n
  n += 1
  cap = ball.colliders.add().capsule
  cap.radius, cap.length = 0.5, 1.0
  thrust = pitm.forces.add(name='ball_thrust', body='ball', strength=1.0).thruster
  thrust.SetInParent()

  # make piggy
  piggy = pitm.bodies.add(name='piggy', mass=1)
  body_idx['piggy'] = n
  n += 1
  box = piggy.colliders.add().box
  dims = box.halfsize
  dims.x, dims.y, dims.z = .5, .5, .5
  # add force (3D)
  thrust = pitm.forces.add(name='piggy_thrust', body='piggy', strength=1.0).thruster
  thrust.SetInParent()

  # make players
  players = []
  for i in range(1,n_players+1):
    p_str = 'p%d'%i
    players.append(p_str)
    player = pitm.bodies.add(name=p_str, mass=1)
    body_idx['p%d'%i] = n
    n += 1
    box = player.colliders.add().box
    l = 1.0
    dims = box.halfsize
    dims.x, dims.y, dims.z = l/2, l/2, l/2
    cap.radius, cap.length = 0.5, 1
    if frozen_players:
      player.frozen.all = True
    else:
      # add force (3D) since players can move around
      thrust = pitm.forces.add(name='p%d_thrust'%i, body='p%d'%i, strength=1.0).thruster
      thrust.SetInParent()

  # make ground
  ground = pitm.bodies.add(name='ground')
  body_idx['ground'] = n
  n += 1
  ground.frozen.all = True
  plane = ground.colliders.add().plane
  plane.SetInParent()  # for setting an empty oneof

  if walls:
    # square walls, 30mx30m, 1m thick, 6m tall
    wall_length = 30
    wall_thickness = 0.5
    wall_height = 6
    for i in range(1,5):
      wall = pitm.bodies.add(name='wall%d'%i, mass=10)
      body_idx['wall_%d'%i] = n
      n += 1
      wall.frozen.all = True
      wall_box = wall.colliders.add().box
      dims = wall_box.halfsize
      if i<3:
        dims.x, dims.y, dims.z = wall_length/2, wall_thickness/2, wall_height/2
      else:
        dims.x, dims.y, dims.z = wall_thickness/2, wall_length/2, wall_height/2
    
  pitm.gravity.z = -9.8
  pitm.friction = friction
  pitm.elasticity = 1.
  pitm.angular_damping = -1.0

  pitm_sys = brax.System(pitm)

  # default starting positions
  default_qp = pitm_sys.default_qp()
  default_qp.pos[body_idx['piggy'],0] = 20  # move piggy init pos
  r = player_radius # starting distance of each player from ball
  t = np.linspace(-np.pi, np.pi, n_players+1)
  dx, dy = r*np.cos(t), r*np.sin(t)
  for i in range(n_players):
    default_qp.pos[body_idx['p1']+i] += np.array([dx[i], dy[i], 0.])
  if ball_init is 'near_p1':
    default_qp.pos[body_idx['ball'], 0] = default_qp.pos[body_idx['p1'],0] + 2
  else:
    default_qp.pos[body_idx['ball'],:2] = ball_init
  if walls:
    default_qp.pos[-1] = jp.array([15, 0, 0])
    default_qp.pos[-2] = jp.array([-15, 0, 0])
    default_qp.pos[-3] = jp.array([0, 15, 0])
    default_qp.pos[-4] = jp.array([0, -15, 0])

  if output_path:
      original_stdout = sys.stdout # Save a reference to the original standard output
      with open(join(output_path, 'config.py'), 'w') as f:
          sys.stdout = f
          print("_SYSTEM_CONFIG = \"\"\"\n", pitm_sys.config, "\n\"\"\"", 
          '\n\nbody_idx = ', body_idx,
          '\n\nn_players = %d'%n_players)
          sys.stdout = original_stdout # Reset the standard output to its original value
        
  # make group_action_shapes
  actions_per_player = 1
  group_action_shapes = make_group_action_shapes(players, actions_per_player)

  # squash together extra useful stuff
  kwargs = dict(
      body_idx=body_idx,
      n_players=n_players,
      group_action_shapes=group_action_shapes,
  )

  return pitm, pitm_sys, default_qp, kwargs


def save_print_as_txt(var, filename, output_path):
  """
  Save the config to a text file.
  """
  original_stdout = sys.stdout
  with open(join(output_path, filename+'.txt'), 'w') as f:
    sys.stdout = f
    print(var)
    sys.stdout = original_stdout # Reset the standard output to its original value

def update_best_params(episode_reward, num_steps, params, metrics, output_path):
  label = 'ep=%.2e_R=%.2e' % (num_steps, episode_reward)
  label = label[:3] + label[7:11] + 'x' + label[3:7] + label[11:] # list files in time order
  path = join(output_path, label)
  if not os.path.exists(path):
    os.mkdir(path)
  model.save_params(join(path, 'params'), params)
  model.save_params(join(path, 'metrics'), metrics)
  return path

def visualize_trajectory(jits, 
                         params, 
                         len_traj=20000, 
                         seed=0, 
                         output_path=None, 
                         rewards_plot=True, 
                         do_not_plot=[],
                         recurrent=False,):
  """
  Visualize the trajectory of the system.
  """
  (env, jit_env_reset, jit_env_step, jit_inference_fn) = jits
  rollout = []
  rng = jax.random.PRNGKey(seed=seed)
  state = jit_env_reset(rng=rng)
  if recurrent:
    len_hidden = recurrent_memory_size # TODO: make not hard coded, add in assertion to check it
    hidden_state = jnp.zeros(len_hidden) # hard coded - naughty
  for _ in range(len_traj):
    rollout.append(state)
    act_rng, rng = jax.random.split(rng)
    if recurrent:
      print('hidden_state: ',hidden_state, jax.tree_utils.tree_map(lambda x: x.shape, hidden_state))
      act, hidden_state = jit_inference_fn(params, state.obs, hidden_state, act_rng)
    if not recurrent:
      act = jit_inference_fn(params, state.obs, act_rng)
    state = jit_env_step(state, act)
    if state.done: # end traj if traj ends
      print('Termination condition reached')
      break
  print('Num timesteps: %d, %d seconds' % (len(rollout), round(len(rollout)*env.sys.config.dt)))

  render_path = join(output_path, 'render_seed=%02d.html'%seed) if output_path \
                else '/content/tmp/render_seed=%02d.html'%seed 
  html.save_html(render_path, env.sys, [s.qp for s in rollout])

  if rewards_plot:
    r_keys = list(state.metrics.keys())
    r_plots = state.metrics # will write over
    for key in r_keys:
      r_plots[key] = []
    r_plots['overall_reward'] = []

    for state in rollout:
      for key in r_keys:
        r_plots[key].append(state.metrics[key])
      r_plots['overall_reward'].append(state.reward)

    r_keys.append('overall_reward')
    # rewards
    fig, ax = plt.subplots(figsize=(12,8))
    legend = []
    num_plots_per_sec = round(1/env.sys.config.dt)
    for key in r_keys:
      if key not in do_not_plot:
        data = r_plots.pop(key)
        legend.append(key)
        if key == 'overall_reward':
          ax.plot(np.linspace(0,len(data)/num_plots_per_sec, len(data)), data, '--')
        else:
          ax.plot(np.linspace(0,len(data)/num_plots_per_sec, len(data)), data)
        print(key, data[-1])
    ax.legend(legend)
    fig_path = join(output_path, 'rewards_seed=%02d.jpg'%seed) if output_path \
                else '/content/tmp/rewards_seed=%02d.jpg'%seed 
    fig.savefig(fig_path)
    fig.savefig(fig_path[:-3] + 'pdf')
    return rollout, render_path, fig_path
  
  return rollout, render_path

def list_except_idx(idx, list):
  x = []
  for i in range(idx):
    x.append(list[i])
  for i in range(idx+1, len(list)):
    x.append(list[i])
  return x

def sample_static_policy(env, rng):
  policies = env.static_agent_params
  rng, rng_agent = jp.random_split(rng)
  agent_idx = jax.random.randint(rng_agent, (1,), 0, policies['num_policies']).astype(int)
  # NN params
  agents_params = []
  for j in range(2): # two agents
    agent_params = {}
    for i in range(5):
      agent_params['hidden_%d'%i] = dict(
          kernel=jnp.squeeze(policies['layers'][j]['hidden_%d'%i][agent_idx,:-1,:]),
          bias=jnp.squeeze(policies['layers'][j]['hidden_%d'%i][agent_idx,-1,:]))
    agents_params.append(agent_params)
  params = dict(policy=agents_params)
  # normalizer
  normalizer = policies['normalizer']
  params['normalizer'] = tuple([normalizer['steps'][agent_idx].squeeze(), 
                                normalizer['mean'][agent_idx].squeeze(), 
                                normalizer['variance'][agent_idx].squeeze()])
  return params, agent_idx, rng

def get_total_count(counters):
  """For counting how many steps are taken for each static policy in training"""
  total_count = counters[0]
  for counter in counters[1:]:
    total_count += counter
  return jnp.sum(total_count, axis=0)