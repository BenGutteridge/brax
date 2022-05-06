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


class VelCap(env.Env):
  """
  Experimenting with an env that has high accelerations but also speed caps.
  """

  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.sys.default_qp()
    qp.pos[1,0] = 20 # move piggy init pos
    qp.pos[2,1] = 3 # move p1 init pos
    qp.pos[3,1] = -3 # move p2 init pos
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'p1_ball_reward': zero,
        'p2_ball_reward': zero,
        'piggy_ball_reward': zero,
        'piggy_touch_ball_reward': zero,
        'ctrl_reward': zero,
        'contact_reward': zero,
        'survive_reward': zero,
        # 'piggy_action': jp.zeros(3),
        # 'player_actions': jp.zeros(3*2),
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
    desired_vel = 1.0             # desired speed of 1m/s
    desired_vel *= vec_piggy_ball # desired velocity vector
    piggy_acc = (desired_vel - state.qp.vel[1,:2]) / self.sys.config.dt # acceleration vector

    # Generating player actions
    act_is_vel = True
    if act_is_vel:
      # Generate force for players: F = m*(v-u)/dt
      desired_vel = action[2:4]
      p1_acc = (desired_vel - state.qp.vel[2,:2]) / self.sys.config.dt
      desired_vel = action[4:]
      p2_acc = (desired_vel - state.qp.vel[3,:2]) / self.sys.config.dt
    else: # use actions as forces directly
      p1_acc, p2_acc = action[2:4], action[4:]

    # Let players apply thrust to the ball
    p1_pos_before, p2_pos_before = state.qp.pos[2,:2], state.qp.pos[3,:2]
    p1_ball_vec = ball_pos_before - p1_pos_before
    p2_ball_vec = ball_pos_before - p2_pos_before
    p1_ball_dist_before = norm(p1_ball_vec)
    p2_ball_dist_before = norm(p2_ball_vec)
    p1_ball_vec /= p1_ball_dist_before # magnitude 1 vectors for exerting thrust on ball
    p2_ball_vec /= p2_ball_dist_before
    # get force magnitude multiplier from action,
    # scale down force based on distance from ball
    ball_thrusters = True
    if ball_thrusters:
      max_dist = 3. # max distance from ball that can still exert force
      p1_force_mult = action[0] * (1 - (p1_ball_dist_before/max_dist))
      p2_force_mult = action[1] * (1 - (p2_ball_dist_before/max_dist)) # linear scale
      p1_force_mult = jp.clip(p1_force_mult, 0, jp.inf)
      p2_force_mult = jp.clip(p2_force_mult, 0, jp.inf)
      # get acceleration vectors
      p1_ball_acc = p1_force_mult * p1_ball_vec
      p2_ball_acc = p2_force_mult * p2_ball_vec
      ball_acc = p1_ball_acc + p2_ball_acc
    else:
      ball_acc = jp.zeros(2)

    # Update step 
    ball_act = jp.concatenate([ball_acc, jp.zeros(1)])
    piggy_act = jp.concatenate([piggy_acc, jp.zeros(1)])
    player_act = jp.concatenate([p1_acc, jp.zeros(1), 
                                  p2_acc, jp.zeros(1)])
    act = jp.concatenate([ball_act, piggy_act, player_act])
    qp, info = self.sys.step(state.qp, act)
    obs = self._get_obs(qp, info)

    #### REWARDS ####
    # convention: all terms positive, subtract terms labelled 'cost', add 'reward'
    # terms:
    #   `p1_ball_reward`, `p2_ball_reward`  : +ve rewards for players approaching ball
    #   `piggy_ball_reward`                 : +ve reward for piggy moving away from ball
    #   `piggy_touch_ball_cost`             : -ve cost for piggy touching ball, end episode
    #   `ctrl_cost`                         : -ve cost for control (input forces)
    #   `contact_cost`                      : -ve cost for contact
    #   `survive_reward`                    : +ve fixed reward for episode not ending

    # Each player move towards ball, small reward
    scale = 10.0
    p1_pos_after, p2_pos_after = qp.pos[2,:2], qp.pos[3,:2]
    ball_pos_after = qp.pos[0,:2]
    p1_ball_dist_after = norm(ball_pos_after - p1_pos_after)
    p2_ball_dist_after = norm(ball_pos_after - p2_pos_after)
    # +ve change, towards ball
    p1_ball_dist_change = (p1_ball_dist_before - p1_ball_dist_after) / self.sys.config.dt
    p2_ball_dist_change = (p2_ball_dist_before - p2_ball_dist_after) / self.sys.config.dt
    p1_ball_reward, p2_ball_reward = p1_ball_dist_change , p2_ball_dist_change
    p1_ball_reward *= scale
    p2_ball_reward *= scale

    # Ball move away from piggy, reward
    scale = 30.0
    piggy_pos_after = qp.pos[1,:2]
    piggy_ball_dist_after = norm(ball_pos_after - piggy_pos_after)
    # +ve means piggy is further away from ball
    piggy_ball_dist_change = (piggy_ball_dist_after - piggy_ball_dist_before) / self.sys.config.dt
    piggy_ball_reward = piggy_ball_dist_change * scale
    
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
    reward = (p1_ball_reward + p2_ball_reward +
              piggy_ball_reward - piggy_touch_ball_cost - 
              ctrl_cost - contact_cost + survive_reward)

    state.metrics.update(
        p1_ball_reward=p1_ball_reward,
        p2_ball_reward=p2_ball_reward,
        piggy_ball_reward=piggy_ball_reward,
        piggy_touch_ball_reward=-1*piggy_touch_ball_cost,
        ctrl_reward=-1*ctrl_cost,
        contact_reward=-1*contact_cost,
        survive_reward=survive_reward,
        # piggy_action=piggy_act,
        # player_actions=player_act,
    )

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return 6 # 2 each for each player to exert on themselves, 1 per player to exert on ball

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

dynamics_mode: "pbd"

""" 
