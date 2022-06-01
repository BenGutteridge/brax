import brax
import sys
import numpy as np
import brax.jumpy as jp
import os
from os.path import join
from brax.io import model



def make_config(n_players=2, 
                walls=False, 
                output_path=False, 
                frozen_players=False,
                friction=0.,
                player_radius=3.,
                ball_init=[0.,0.],
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
      wall = pitm.bodies.add(name='wall%d'%i)
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
  default_qp.pos[body_idx['ball'],:2] = ball_init
  default_qp.pos[body_idx['piggy'],0] = 20  # move piggy init pos
  r = player_radius # starting distance of each player from ball
  t = np.linspace(-np.pi, np.pi, n_players+1)
  dx, dy = r*np.cos(t), r*np.sin(t)
  for i in range(n_players):
    default_qp.pos[body_idx['p1']+i] += np.array([dx[i], dy[i], 0.])
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

  return pitm, pitm_sys, default_qp


def save_config_txt(config, output_path):
  """
  Save the config to a text file.
  """
  original_stdout = sys.stdout
  with open(join(output_path, 'config.txt'), 'w') as f:
    sys.stdout = f
    print(config)
    sys.stdout = original_stdout # Reset the standard output to its original value

def update_best_params(episode_reward, num_steps, params, metrics, output_path):
  label = 'ep=%.2e_R=%.2e' % (num_steps, episode_reward)
  label = label[:3] + label[7:11] + 'x' + label[3:7] + label[11:] # list files in time order
  path = join(output_path, label)
  os.mkdir(path)
  model.save_params(join(path, 'params'), params)
  model.save_params(join(path, 'metrics'), metrics)
  return path