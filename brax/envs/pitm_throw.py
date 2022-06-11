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

"""Trains an ant to run in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import env
from brax.jumpy import safe_norm as norm
from brax.ben_utils.utils import make_config, list_except_idx
import copy

from collections import OrderedDict as odict
from typing import Dict, Any, Callable, Tuple, Optional, Union
from jax import numpy as jnp


class PITM_Throw(env.Env):
  """
  An env that has stationary players that can launch ball *when nearby* at a constant velocity, continuous angle choice.
  """

  def __init__(self, **kwargs):    
    # make config
    _, _, self.default_qp, args = make_config(n_players=5, 
                                              walls=kwargs.pop('walls', True),
                                              frozen_players=True,
                                              output_path='.',
                                              friction=0.,
                                              player_radius=7.,
    )
    from config import _SYSTEM_CONFIG as config
    super().__init__(config=config, **kwargs)
    # adaptations to MAPPO
    self.idx=args['body_idx']
    self.n_players=args['n_players']
    self.group_action_shapes=args['group_action_shapes']
    self.is_multiagent = True
    self.reward_shape = (len(
        self.group_action_shapes),)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state (same every time, currently)"""
    qp = self.default_qp
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    zero = jp.float32(0) # consistent shapes
    # making sure size of reward is same as number of agents
    reward = jnp.zeros(self.reward_shape)
    done = jp.float32(0)  # ensure done is a scalar
    self.player_poses = [qp.pos[self.idx['p%d'%i],:2] for i in range(1, self.n_players+1)]
    self.previous_player_idx = self.idx['p1']

    metrics = {
        'piggy_touch_ball_reward': zero,
        'ctrl_reward': zero,
        'survive_reward': zero,
        'num_passes': zero,
        'ball_passing_reward': zero,
        'agg_ball_passing_reward': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, 
           state: env.State, 
           action: jp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None ) -> env.State:
    """Run one timestep of the environment's dynamics."""

    del normalizer_params, extra_params

    # Generating piggy action
    ball_pos_before = state.qp.pos[self.idx['ball'],:2]
    piggy_pos_before = state.qp.pos[self.idx['piggy'],:2]
    vec_piggy_ball = ball_pos_before - piggy_pos_before
    piggy_ball_dist_before = norm(vec_piggy_ball)
    # check that this won't make velocity go too high
    vec_piggy_ball /= piggy_ball_dist_before # normalize
    # Generate force for piggy: F = m*(v-u)/dt, where v is desired_vel, direction -> ball.
    desired_vel = 2.0             # desired speed
    desired_vel *= vec_piggy_ball # desired velocity vector
    piggy_acc = (desired_vel - state.qp.vel[self.idx['piggy'],:2]) / self.sys.config.dt # acceleration vector

    # Let players apply thrust to the ball
    num_actions_per_player = 2 # x,y acc
    force_mult = 25.
    ball_acc = jp.zeros(2) # x, y
    for i in range(self.n_players):
      player_pos = self.player_poses[i]
      player_ball_dist = norm(player_pos - ball_pos_before)
      acc_x = action[num_actions_per_player * i] * (1/player_ball_dist**2) * force_mult
      acc_y = action[num_actions_per_player * i + 1] * (1/player_ball_dist**2) * force_mult
      ball_acc += jnp.array([acc_x, acc_y])
    
    # ball drag
    visc = 0.1 # 1.81e-5 # viscosity of air
    ball_r = self.sys.config.bodies[0].colliders[0].capsule.radius
    ball_drag = 6 * jp.pi * ball_r * visc * state.qp.vel[0,:2]
    ball_acc -= ball_drag

    # Update step 
    ball_act = jp.concatenate([ball_acc, jp.zeros(1)])
    piggy_act = jp.concatenate([piggy_acc, jp.zeros(1)])
    act = jp.concatenate([ball_act, piggy_act])
    qp, info = self.sys.step(state.qp, act)
    obs = self._get_obs(qp, info)

    # New nearest player
    ball_pos_after = qp.pos[self.idx['ball'],:2]
    nearest_player_idx = jnp.argmin(norm(jp.array(self.player_poses) - ball_pos_after, axis=1))

    #### REWARDS ####
    # convention: all terms positive, subtract terms labelled 'cost', add 'reward'
    # terms:
    #   `ctrl_cost`                         : -ve cost for control (input forces)
    #   `survive_reward`                    : +ve fixed reward for episode not ending
    #   `piggy_touch_ball_reward`           : big -ve reward for piggy reaching ball 
    
    # Piggy reach ball, big cost, end episode
    scale = 1000.
    eps = 1.30 # minimum distance between ball and piggy centres ((1+root2)/2 + a bit)
    piggy_pos_after = qp.pos[self.idx['piggy'],:2]
    piggy_ball_dist_after = norm(ball_pos_after - piggy_pos_after)
    piggy_touch_ball_cost = (piggy_ball_dist_after < eps) * scale
    done = jp.where(piggy_ball_dist_after < eps, jp.float32(1), jp.float32(0)) # end if piggy touches ball

    # Large fixed cost for BALL getting outside walls
    fixed_cost, scale = 1000, 1
    out_of_bounds_cost = 0.
    # for pos in [ball_pos_after]:
    #   out_of_bounds_cost += jp.amax(jp.where(abs(pos) > 16, jp.float32(1), jp.float32(0)))
    # out_of_bounds_cost *= fixed_cost * scale 
    # done = jp.where(out_of_bounds_cost > 1, jp.float32(1), jp.float32(0)) # if, then, else

    # Reward for 'ball passed' - nearest player changing
    scale = 100 * state.metrics['num_passes']
    ball_passed_reward = jp.where(nearest_player_idx != self.previous_player_idx, jp.float32(1), jp.float32(0))
    ball_passed_reward *= scale

    # Reward for ball passing from current player to one of the others
    scale = 20
    other_player_poses = [self.player_poses[i] for i in range(self.n_players) if i != self.previous_player_idx]
    ball_player_deltas = jp.array([norm(ball_pos_before - pos) - norm(ball_pos_after - pos) for pos in other_player_poses])
    # +ve is towards player
    ball_passing_reward = jnp.max(ball_player_deltas) / self.sys.config.dt * scale

    # Aggressively reward shaped ball passing
    scale = 1
    """
    1. Get pose of 'next' player
    2. Get vector from ball towards next player
    3. Give a reward for action based on inner product with vector from ball to next player
    """
    next_player_idx = (self.previous_player_idx + 1) % self.n_players
    next_player_pose = self.player_poses[next_player_idx]
    ball_next_player_vec = next_player_pose - ball_pos_before
    ball_next_player_vec /= norm(ball_next_player_vec)
    agg_ball_passing_reward = jnp.dot(ball_next_player_vec, ball_acc) * scale

    # standard stuff -- contact cost, survive reward, control cost
    ctrl_cost = 0. # .5 * jp.sum(jp.square(action)) # let's encourage movement
    survive_reward = jp.float32(1)

    # total reward
    costs = ctrl_cost + piggy_touch_ball_cost + out_of_bounds_cost
    reward = agg_ball_passing_reward + ball_passing_reward + ball_passed_reward + survive_reward - costs
    reward *= jp.ones_like(state.reward) # make sure it's the right shape - DecPOMDP so same reward for all agents

    state.metrics.update(
        piggy_touch_ball_reward=-1*piggy_touch_ball_cost,
        ctrl_reward=-1*ctrl_cost,
        survive_reward=survive_reward,
        num_passes=state.metrics['num_passes'] + jp.where(ball_passed_reward > 0, jp.float32(1), jp.float32(0)),
        ball_passing_reward=ball_passing_reward,
        agg_ball_passing_reward=agg_ball_passing_reward,
    )

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return self.n_players * 2 # x, y forces exerted on ball by each player

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe ant body position and velocities."""

    #### OBSERVATIONS ####
    all_body_pos = qp.pos[:2,:2].flatten() # x,y positions of ball and piggy (4,)
    player_poses = qp.pos[2:2+self.n_players, :2].flatten() # x,y positions of players (n_players*2)
    all_body_vel = qp.vel[:2,:2].flatten() # x,y velocities of ball and piggy (4,)
    # ball_ang  = qp.ang[0]                   # ball angular velocities (3,)

    return jp.concatenate([all_body_pos] + [all_body_vel] + [player_poses])