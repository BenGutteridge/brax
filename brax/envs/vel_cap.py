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
        'piggy_action': jp.zeros(3),
        'player_actions': jp.zeros(3*2),
        # 'action': jp.zeros(self.action_size),
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
    piggy_acc = 1.0 # base acceleration of 1m/s^2
    piggy_acc *= vec_piggy_ball # convert into acceleration vector
    piggy_acc_x, piggy_acc_y = piggy_acc

    # Ensure piggy thrust doesn't cause it to exceed max velocity
    piggy_max_vel = 3.0
    next_x_vel, next_y_vel = state.qp.vel[0,:2] + piggy_acc * self.sys.config.dt # v = v0 + a * dt
    piggy_acc_x = jp.where(jp.abs(next_x_vel) < piggy_max_vel,  # if
                            piggy_acc_x,                        # then
                            jp.float32(0))                      # else
    piggy_acc_y = jp.where(jp.abs(next_y_vel) < piggy_max_vel,  # if
                            piggy_acc_y,                        # then
                            jp.float32(0))                      # else
    piggy_acc = jp.array([piggy_acc_x, piggy_acc_y])

    # Update step 
    piggy_act = jp.concatenate([piggy_acc, jp.zeros(1)])
    player_act = jp.concatenate([action[:2], jp.zeros(1), 
                                  action[2:], jp.zeros(1)])
    act = jp.concatenate([piggy_act, player_act])
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
    scale = 5.0
    p1_pos_before, p1_pos_after = state.qp.pos[2,:2], qp.pos[2,:2]
    p2_pos_before, p2_pos_after = state.qp.pos[3,:2], qp.pos[3,:2]
    ball_pos_after = qp.pos[0,:2]
    p1_ball_dist_before = norm(ball_pos_before - p1_pos_before)
    p1_ball_dist_after = norm(ball_pos_after - p1_pos_after)
    p2_ball_dist_before = norm(ball_pos_before - p2_pos_before)
    p2_ball_dist_after = norm(ball_pos_after - p2_pos_after)
    # +ve change, towards ball
    p1_ball_dist_change = (p1_ball_dist_before - p1_ball_dist_after) / self.sys.config.dt
    p2_ball_dist_change = (p2_ball_dist_before - p2_ball_dist_after) / self.sys.config.dt
    p1_ball_reward, p2_ball_reward = p1_ball_dist_change , p2_ball_dist_change
    p1_ball_reward *= scale
    p2_ball_reward *= scale

    # Ball move away from piggy, reward
    scale = 5.0
    piggy_pos_after = qp.pos[1,:2]
    piggy_ball_dist_after = norm(ball_pos_after - piggy_pos_after)
    # +ve means piggy is further away from ball
    piggy_ball_dist_change = (piggy_ball_dist_after - piggy_ball_dist_before) / self.sys.config.dt
    piggy_ball_reward = piggy_ball_dist_change * scale
    
    # Piggy reach ball, big cost, end episode
    scale = 1000.
    eps = 1.05 # minimum distance between ball and piggy centres
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
        piggy_action=piggy_act,
        player_actions=player_act,
        # action=action
    )

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return 4

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
angular_damping: -0.05
dt: 0.05000000074505806
substeps: 20
frozen {
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
