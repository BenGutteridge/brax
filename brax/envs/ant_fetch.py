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

"""Trains an agent to locomote to a target location."""

from typing import Tuple

import brax
from brax import jumpy as jp
from brax import math
from brax.envs import env
from brax.ben_utils.utils import make_group_action_shapes
from jax import numpy as jnp


class AntFetch(env.Env):
  """Fetch trains an ant to run to a target location."""

  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG
    is_multiagent = False if kwargs.pop('is_not_multiagent', False) else True
    super().__init__(config=config, **kwargs)
    if is_multiagent:
      self.n_agents, self.actuators_per_agent = 2, 4
      players = ['agent_%d' % i for i in range(self.n_agents)]
      self.group_action_shapes = make_group_action_shapes(players, self.actuators_per_agent)
      self.is_multiagent = True
      self.reward_shape = (len(self.group_action_shapes),)
    else: self.reward_shape = 1
    self.target_idx = self.sys.body.index['Target']
    self.torso_idx = self.sys.body.index['$ Torso']
    self.target_radius = 2
    self.target_distance = 15

  def reset(self, rng: jp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    rng, target = self._random_target(rng)
    pos = jp.index_update(qp.pos, self.target_idx, target)
    qp = qp.replace(pos=pos)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    done, zero = jp.zeros(2)
    reward = jnp.zeros(self.reward_shape)
    metrics = {
        'hits': zero,
        'weightedHits': zero,
        'movingToTarget': zero,
        'torsoIsUp': zero,
        'torsoHeight': zero
    }
    info = {'rng': rng}
    return env.State(qp, obs, reward, done, metrics, info)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # small reward for torso moving towards target
    torso_delta = qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
    target_rel = qp.pos[self.target_idx] - qp.pos[self.torso_idx]
    target_dist = jp.norm(target_rel)
    target_dir = target_rel / (1e-6 + target_dist)
    moving_to_target = .1 * jp.dot(torso_delta, target_dir)

    # small reward for torso being up
    up = jp.array([0., 0., 1.])
    torso_up = math.rotate(up, qp.rot[self.torso_idx])
    torso_is_up = .1 * self.sys.config.dt * jp.dot(torso_up, up)

    # small reward for torso height
    torso_height = .1 * self.sys.config.dt * qp.pos[0, 2]

    # big reward for reaching target and facing it
    fwd = jp.array([1., 0., 0.])
    torso_fwd = math.rotate(fwd, qp.rot[self.torso_idx])
    torso_facing = jp.dot(target_dir, torso_fwd)
    target_hit = target_dist < self.target_radius
    target_hit = jp.where(target_hit, jp.float32(1), jp.float32(0))
    weighted_hit = target_hit * torso_facing

    reward = torso_height + moving_to_target + torso_is_up # + weighted_hit
    reward *= jp.ones_like(state.reward)

    state.metrics.update(
        hits=target_hit,
        weightedHits=weighted_hit,
        movingToTarget=moving_to_target,
        torsoIsUp=torso_is_up,
        torsoHeight=torso_height)

    # teleport any hit targets
    rng, target = self._random_target(state.info['rng'])
    target = jp.where(target_hit, target, qp.pos[self.target_idx])
    pos = jp.index_update(qp.pos, self.target_idx, target)
    qp = qp.replace(pos=pos)
    state.info.update(rng=rng)
    return state.replace(qp=qp, obs=obs, reward=reward)

  @property
  def action_size(self):
    return self.n_agents * self.actuators_per_agent

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Egocentric observation of target and the ant's body."""
    torso_fwd = math.rotate(jp.array([1., 0., 0.]), qp.rot[self.torso_idx])
    torso_up = math.rotate(jp.array([0., 0., 1.]), qp.rot[self.torso_idx])

    v_inv_rotate = jp.vmap(math.inv_rotate, include=(True, False))

    pos_local = qp.pos - qp.pos[self.torso_idx]
    pos_local = v_inv_rotate(pos_local, qp.rot[self.torso_idx])
    vel_local = v_inv_rotate(qp.vel, qp.rot[self.torso_idx])

    target_local = pos_local[self.target_idx]
    target_local_mag = jp.reshape(jp.norm(target_local), -1)
    target_local_dir = target_local / (1e-6 + target_local_mag)

    pos_local = jp.reshape(pos_local, -1)
    vel_local = jp.reshape(vel_local, -1)

    contact_mag = jp.sum(jp.square(info.contact.vel), axis=-1)
    contacts = jp.where(contact_mag > 0.00001, 1, 0)

    return jp.concatenate([
        torso_fwd, torso_up, target_local_mag, target_local_dir, pos_local,
        vel_local, contacts
    ])

  def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns a target location in a random circle on xz plane."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    dist = self.target_radius + self.target_distance * jp.random_uniform(rng1)
    ang = jp.pi * 2. * jp.random_uniform(rng2)
    target_x = dist * jp.cos(ang)
    target_y = dist * jp.sin(ang)
    target_z = 1.0
    target = jp.array([target_x, target_y, target_z]).transpose()
    return rng, target


_SYSTEM_CONFIG = """
  bodies {
    name: "$ Torso"
    colliders {
      capsule {
        radius: 0.25
        length: 0.5
        end: 1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 10
  }
  bodies {
    name: "Aux 1"
    colliders {
      rotation { x: 90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 4"
    colliders {
      rotation { x: 90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 2"
    colliders {
      rotation { x: 90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 7"
    colliders {
      rotation { x: 90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 3"
    colliders {
      rotation { x: -90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 10"
    colliders {
      rotation { x: -90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 4"
    colliders {
      rotation { x: -90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 13"
    colliders {
      rotation { x: -90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Ground"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }

  bodies {
    name: "Target"
    colliders { sphere { radius: 2 }}
    frozen { all: true }
  }

  joints {
    name: "$ Torso_Aux 1"
    parent_offset { x: 0.2 y: 0.2 }
    child_offset { x: -0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 1"
    angle_limit { min: -30.0 max: 30.0 }
    rotation { y: -90 }
    angular_damping: 20
  }
  joints {
    name: "Aux 1_$ Body 4"
    parent_offset { x: 0.1 y: 0.1 }
    child_offset { x: -0.2 y: -0.2 }
    parent: "Aux 1"
    child: "$ Body 4"
    rotation: { z: 135 }
    angle_limit {
      min: 30.0
      max: 70.0
    }
    angular_damping: 20
  }
  joints {
    name: "$ Torso_Aux 2"
    parent_offset { x: -0.2 y: 0.2 }
    child_offset { x: 0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 2"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "Aux 2_$ Body 7"
    parent_offset { x: -0.1 y: 0.1 }
    child_offset { x: 0.2 y: -0.2 }
    parent: "Aux 2"
    child: "$ Body 7"
    rotation { z: 45 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "$ Torso_Aux 3"
    parent_offset { x: -0.2 y: -0.2 }
    child_offset { x: 0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 3"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "Aux 3_$ Body 10"
    parent_offset { x: -0.1 y: -0.1 }
    child_offset {
      x: 0.2
      y: 0.2
    }
    parent: "Aux 3"
    child: "$ Body 10"
    rotation { z: 135 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "$ Torso_Aux 4"
    parent_offset { x: 0.2 y: -0.2 }
    child_offset { x: -0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 4"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "Aux 4_$ Body 13"
    parent_offset { x: 0.1 y: -0.1 }
    child_offset { x: -0.2 y: 0.2 }
    parent: "Aux 4"
    child: "$ Body 13"
    rotation { z: 45 }
    angle_limit { min: 30.0 max: 70.0 }
    angular_damping: 20
  }
  actuators {
    name: "$ Torso_Aux 1"
    joint: "$ Torso_Aux 1"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "Aux 1_$ Body 4"
    joint: "Aux 1_$ Body 4"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "$ Torso_Aux 2"
    joint: "$ Torso_Aux 2"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "Aux 2_$ Body 7"
    joint: "Aux 2_$ Body 7"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "$ Torso_Aux 3"
    joint: "$ Torso_Aux 3"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "Aux 3_$ Body 10"
    joint: "Aux 3_$ Body 10"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "$ Torso_Aux 4"
    joint: "$ Torso_Aux 4"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "Aux 4_$ Body 13"
    joint: "Aux 4_$ Body 13"
    strength: 350.0
    torque {}
  }
  friction: 1.0
  gravity { z: -9.8 }
  angular_damping: -0.05
  collide_include {
    first: "$ Torso"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 4"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 7"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 10"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 13"
    second: "Ground"
  }
  dt: 0.05
  substeps: 10
  dynamics_mode: "pbd"
  """