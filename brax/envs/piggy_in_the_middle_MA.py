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


class PITM_MA(env.Env):
  """
  Experimenting with an env that has high accelerations but also speed caps.
  """

  def __init__(self, legacy_spring=False, **kwargs):
    try:
      self.walls = kwargs.pop('walls')
    except:
      self.walls = False
    config = _SYSTEM_CONFIG_WALLS if self.walls else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.sys.default_qp()
    qp.pos[1,0] = 20                  # move piggy init pos
    qp.pos[2,1] = 3                   # move p1 init pos
    qp.pos[3,1] = -3                  # move p2 init pos
    qp.pos[4,:2] = jp.array([-3, 0])  # move p3 init pos
    if self.walls:
      qp.pos[-1] = jp.array([15, 0, 0])
      qp.pos[-2] = jp.array([-15, 0, 0])
      qp.pos[-3] = jp.array([0, 15, 0])
      qp.pos[-4] = jp.array([0, -15, 0])
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'p1_ball_reward': zero,
        'p2_ball_reward': zero,
        'p3_ball_reward': zero,
        'player_separation_reward': zero,
        'out_of_bounds_reward': zero,
        'piggy_ball_reward': zero,
        'piggy_ball_static_reward': zero,
        'piggy_touch_ball_reward': zero,
        'ctrl_reward': zero,
        'contact_reward': zero,
        'survive_reward': zero,
        'ball_thrust_x': zero,
        'ball_thrust_y': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""

    # Generating piggy action
    ball_pos_before = state.qp.pos[0,:2]
    piggy_pos_before = state.qp.pos[1,:2]
    vec_piggy_ball = ball_pos_before - piggy_pos_before
    piggy_ball_dist_before = norm(vec_piggy_ball)
    # check that this won't make velocity go too high
    vec_piggy_ball /= piggy_ball_dist_before # normalize
    # Generate force for piggy: F = m*(v-u)/dt, where v is desired_vel, direction -> ball.
    desired_vel = 2.0             # desired speed
    desired_vel *= vec_piggy_ball # desired velocity vector
    piggy_acc = (desired_vel - state.qp.vel[1,:2]) / self.sys.config.dt # acceleration vector

    # Generating player actions
    act_is_vel = True
    if act_is_vel:
      vel_mult = 10.
      # Generate force for players: F = m*(v-u)/dt
      desired_vel = vel_mult * action[:2]
      p1_acc = (desired_vel - state.qp.vel[2,:2]) / self.sys.config.dt
      desired_vel = vel_mult * action[2:4]
      p2_acc = (desired_vel - state.qp.vel[3,:2]) / self.sys.config.dt
      desired_vel = vel_mult * action[4:6]
      p3_acc = (desired_vel - state.qp.vel[4,:2]) / self.sys.config.dt
    else: # use actions as forces directly
      p1_acc, p2_acc, p3_acc = action[:2], action[2:4], action[4:6]

    # Let players apply thrust to the ball
    # p1
    p1_pos_before = state.qp.pos[2,:2]
    p1_ball_vec = ball_pos_before - p1_pos_before
    p1_ball_dist_before = norm(p1_ball_vec)
    p1_ball_vec /= p1_ball_dist_before # magnitude 1 vectors for exerting thrust on ball
    # p2
    p2_pos_before = state.qp.pos[3,:2]
    p2_ball_vec = ball_pos_before - p2_pos_before
    p2_ball_dist_before = norm(p2_ball_vec)
    p2_ball_vec /= p2_ball_dist_before
    # p3
    p3_pos_before = state.qp.pos[4,:2]
    p3_ball_vec = ball_pos_before - p3_pos_before
    p3_ball_dist_before = norm(p3_ball_vec)
    p3_ball_vec /= norm(p3_ball_vec)
    # get force magnitude multiplier from action,
    # scale down force based on distance from ball
    ball_thrusters = True
    if ball_thrusters:
      force_multiplier = 10.
      p1_force_mult = force_multiplier*(action[-1]/p1_ball_dist_before)**2 # inverse sq dropoff
      p2_force_mult = force_multiplier*(action[-2]/p2_ball_dist_before)**2
      p3_force_mult = force_multiplier*(action[-3]/p3_ball_dist_before)**2
      # get acceleration vectors
      p1_ball_acc = p1_force_mult * p1_ball_vec
      p2_ball_acc = p2_force_mult * p2_ball_vec
      p3_ball_acc = p3_force_mult * p3_ball_vec
      ball_acc = p1_ball_acc + p2_ball_acc + p3_ball_acc
    else:
      ball_acc = jp.zeros(2)
    # ball drag (is this basically negigible?)
    visc = 1.81e-5 # viscosity of air
    ball_r = self.sys.config.bodies[0].colliders[0].capsule.radius
    ball_drag = 6 * jp.pi * ball_r * visc * state.qp.vel[0,:2]
    ball_acc -= ball_drag


    # Update step 
    ball_act = jp.concatenate([ball_acc, jp.zeros(1)])
    piggy_act = jp.concatenate([piggy_acc, jp.zeros(1)])
    player_act = jp.concatenate([p1_acc, jp.zeros(1), 
                                  p2_acc, jp.zeros(1), 
                                  p3_acc, jp.zeros(1)])
    act = jp.concatenate([ball_act, piggy_act, player_act])
    qp, info = self.sys.step(state.qp, act)
    obs = self._get_obs(qp, info)

    #### REWARDS ####
    # convention: all terms positive, subtract terms labelled 'cost', add 'reward'
    # terms:
    #   `p1_ball_reward`, `p2_ball_reward`, `p3_ball_reward`  : +ve rewards for players approaching ball
    #   `piggy_ball_reward`                 : +ve reward for piggy moving away from ball
    #   `piggy_touch_ball_cost`             : -ve cost for piggy touching ball, end episode
    #   `ctrl_cost`                         : -ve cost for control (input forces)
    #   `contact_cost`                      : -ve cost for contact
    #   `survive_reward`                    : +ve fixed reward for episode not ending

    # Each player move towards ball, small reward
    scale = 5.0
    ball_pos_after = qp.pos[0,:2]
    p1_pos_after = qp.pos[2,:2]
    p2_pos_after = qp.pos[3,:2]
    p3_pos_after = qp.pos[4,:2]
    p1_ball_dist_after = norm(ball_pos_after - p1_pos_after)
    p2_ball_dist_after = norm(ball_pos_after - p2_pos_after)
    p3_ball_dist_after = norm(ball_pos_after - p3_pos_after)
    # +ve change, towards ball
    p1_ball_dist_change = (p1_ball_dist_before - p1_ball_dist_after) / self.sys.config.dt
    p2_ball_dist_change = (p2_ball_dist_before - p2_ball_dist_after) / self.sys.config.dt
    p3_ball_dist_change = (p3_ball_dist_before - p3_ball_dist_after) / self.sys.config.dt
    p1_ball_reward, p2_ball_reward, p3_ball_reward = p1_ball_dist_change , p2_ball_dist_change, p3_ball_dist_change
    p1_ball_reward *= scale
    p2_ball_reward *= scale
    p3_ball_reward *= scale

    # Reward for players being far from each other
    scale = 1.
    player_poses = [p1_pos_after, p2_pos_after, p3_pos_after]
    p_dists = jp.float32(0)
    while len(player_poses):
      player_pose = player_poses.pop(0)
      p_dists += jp.sum(jp.array([norm(player_pose - p) for p in player_poses]))
    player_separation_reward = p_dists * scale

    
    # Large fixed cost for player (OR BALL) getting outside walls
    fixed_cost, scale = 1000, 1
    out_of_bounds_cost = 0.
    for pos in [p1_pos_after, p2_pos_after, p3_pos_after, ball_pos_after]:
      out_of_bounds_cost += jp.amax(jp.where(abs(pos) > 16, jp.float32(1), jp.float32(0)))
    out_of_bounds_cost *= fixed_cost * scale 
    done = jp.where(out_of_bounds_cost > 1, jp.float32(1), jp.float32(0)) # if, then, else

    # Ball move away from piggy, reward
    scale = 20.0
    piggy_pos_after = qp.pos[1,:2]
    piggy_ball_dist_after = norm(ball_pos_after - piggy_pos_after)
    # +ve means piggy is further away from ball
    piggy_ball_dist_change = (piggy_ball_dist_after - piggy_ball_dist_before) / self.sys.config.dt
    piggy_ball_reward = piggy_ball_dist_change * scale

    # Ball dist from piggy reward (not change)
    scale = 1.0
    piggy_ball_static_reward = piggy_ball_dist_after * scale
    
    # Piggy reach ball, big cost, end episode
    scale = 1000.
    eps = 1.21 # minimum distance between ball and piggy centres (~(1+root2)/2)
    piggy_touch_ball_cost = (piggy_ball_dist_after < eps) * scale
    done = jp.where(piggy_ball_dist_after < eps, jp.float32(1), jp.float32(0)) # if, then, else

    # standard stuff -- contact cost, survive reward, control cost
    ctrl_cost = 0. # .5 * jp.sum(jp.square(action)) # let's encourage movement
    contact_cost = (0.5 * 1e-3 *
                    jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
    survive_reward = jp.float32(1)

    # total reward
    reward = (p1_ball_reward + p2_ball_reward + p3_ball_reward +
              player_separation_reward -
              out_of_bounds_cost +
              piggy_ball_reward + piggy_ball_static_reward - 
              piggy_touch_ball_cost - 
              ctrl_cost - contact_cost + survive_reward)

    state.metrics.update(
        p1_ball_reward=p1_ball_reward,
        p2_ball_reward=p2_ball_reward,
        p3_ball_reward=p3_ball_reward,
        player_separation_reward=player_separation_reward,
        out_of_bounds_reward=-1*out_of_bounds_cost,
        piggy_ball_reward=piggy_ball_reward,
        piggy_ball_static_reward=piggy_ball_static_reward,
        piggy_touch_ball_reward=-1*piggy_touch_ball_cost,
        ctrl_reward=-1*ctrl_cost,
        contact_reward=-1*contact_cost,
        survive_reward=survive_reward,
        ball_thrust_x=ball_acc[0],
        ball_thrust_y=ball_acc[1],
    )

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return 9 # 3 per player, 2 for their own movement, 1 for exerting force on ball

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe ant body position and velocities."""

    #### OBSERVATIONS ####
    all_body_pos = qp.pos[:-1,:2].flatten() # x,y positions of both players, ball and piggy (8,)
    all_body_vel = qp.vel[:-1,:2].flatten() # x,y velocities of both players, ball and piggy (8,)
    ball_ang  = qp.ang[0]                   # ball angular velocities (3,)

    return jp.concatenate([all_body_pos] + [all_body_vel] + [ball_ang])


_SYSTEM_CONFIG = """
bodies {
  name: "ball"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "piggy"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "p1"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "p2"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "p3"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

 bodies {
  name: "ground"
  colliders {
    plane {
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
elasticity: 1.0
friction: 0.1
gravity {
  z: -9.800000190734863
}
angular_damping: -0.20
dt: 0.05000000074505806
substeps: 20
frozen {
}

forces {
  name: "ball_thrust"
  body: "ball"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "piggy_thrust"
  body: "piggy"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "p1_thrust"
  body: "p1"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "p2_thrust"
  body: "p2"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "p3_thrust"
  body: "p3"
  strength: 1.0
  thruster {
  }
}

dynamics_mode: "pbd"

""" 

_SYSTEM_CONFIG_WALLS = """
bodies {
  name: "ball"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "piggy"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "p1"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "p2"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

bodies {
  name: "p3"
  colliders {
    box {
      halfsize {
        x: 0.5
        y: 0.5
        z: 0.5
      }
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}

 bodies {
  name: "ground"
  colliders {
    plane {
    }
    material {
      elasticity: 1.0
      friction: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}

bodies {
  name: "wall_1"
  colliders {
    box {
      halfsize {
        x: 15.0
        y: 0.25
        z: 3.0
      }
    }
    material {
      elasticity: 1.0
      friction: 10.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 10.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
bodies {
  name: "wall_2"
  colliders {
    box {
      halfsize {
        x: 15.0
        y: 0.25
        z: 3.0
      }
    }
    material {
      elasticity: 1.0
      friction: 10.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 10.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
bodies {
  name: "wall_3"
  colliders {
    box {
      halfsize {
        x: 0.25
        y: 15.0
        z: 3.0
      }
    }
    material {
      elasticity: 1.0
      friction: 10.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 10.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
bodies {
  name: "wall_4"
  colliders {
    box {
      halfsize {
        x: 0.25
        y: 15.0
        z: 3.0
      }
    }
    material {
      elasticity: 1.0
      friction: 10.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 10.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}

elasticity: 1.0
friction: 0.1
gravity {
  z: -9.800000190734863
}
angular_damping: -0.20
dt: 0.05000000074505806
substeps: 20
frozen {
}

forces {
  name: "ball_thrust"
  body: "ball"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "piggy_thrust"
  body: "piggy"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "p1_thrust"
  body: "p1"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "p2_thrust"
  body: "p2"
  strength: 1.0
  thruster {
  }
}

forces {
  name: "p3_thrust"
  body: "p3"
  strength: 1.0
  thruster {
  }
}

dynamics_mode: "pbd"

"""