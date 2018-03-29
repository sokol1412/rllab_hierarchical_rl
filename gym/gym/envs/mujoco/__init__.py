from gym.gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.gym.envs.mujoco.ant import AntEnv
from gym.gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.gym.envs.mujoco.hopper import HopperEnv
from gym.gym.envs.mujoco.walker2d import Walker2dEnv
from gym.gym.envs.mujoco.humanoid import HumanoidEnv
from gym.gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.gym.envs.mujoco.reacher import ReacherEnv
from gym.gym.envs.mujoco.swimmer import SwimmerEnv
from gym.gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.gym.envs.mujoco.pusher import PusherEnv
from gym.gym.envs.mujoco.thrower import ThrowerEnv
from gym.gym.envs.mujoco.striker import StrikerEnv
