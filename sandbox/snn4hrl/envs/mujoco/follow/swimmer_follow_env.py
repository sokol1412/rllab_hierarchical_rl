from sandbox.snn4hrl.envs.mujoco.follow.follow_env import FollowEnv
from sandbox.snn4hrl.envs.mujoco.swimmer_env import SwimmerEnv


class SwimmerFollowEnv(FollowEnv):
    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

