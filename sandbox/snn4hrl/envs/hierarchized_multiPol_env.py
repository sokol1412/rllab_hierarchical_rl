import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from rllab.envs.normalized_env import NormalizedEnv
from sandbox.snn4hrl.envs.mujoco.maze.fast_maze_env import FastMazeEnv
from sandbox.snn4hrl.policies.hier_multi_mlp_policy import GaussianMLPPolicy_multi_hier
from sandbox.snn4hrl.sampler.utils import rollout  # this is a different rollout (option of no reset)


class HierarchizedMultiPoliEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            time_steps_agg=1,
            discrete_actions=True,
            pkl_paths=None,
            json_paths=None,
            npz_paths=None,
            animate=False,
            keep_rendered_rgb=False,
    ):
        """
        :param env: Env to wrap, should have same robot characteristics than env where the policies where pretrained on
        :param time_steps_agg: Time-steps during which one of the policies is executed without external action possible
        :param discrete_actions: whether the policy are applied alone or with a linear combination
        :param pkl_paths: list of paths to pickled pre-training experiment that includes the pre-trained policy
        :param json_paths: list of paths to json of the pre-training experiment. Requires npz_paths of the policy params
        :param npz_paths: only required when using json_paths
        :param keep_rendered_rgb: the returned frac_paths include all rgb images (for plotting video after)
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.time_steps_agg = time_steps_agg
        self.discrete_actions = discrete_actions
        self.animate = animate
        self.keep_rendered_rgb = keep_rendered_rgb
        if json_paths:
            self.low_policy_selector_dim = len(json_paths)
        elif pkl_paths:
            self.low_policy_selector_dim = len(pkl_paths)
        else:
            raise Exception("No path no file given")

        self.low_policy = GaussianMLPPolicy_multi_hier(
            env_spec=env.spec,
            env=env,
            pkl_paths=pkl_paths,
            json_paths=json_paths,
            npz_paths=npz_paths,
            trainable_old=False,
            external_selector=True,
        )

    @property
    @overrides
    def action_space(self):
        selector_dim = self.low_policy_selector_dim
        if self.discrete_actions:
            return spaces.Discrete(selector_dim)  # the action is now just a selection
        else:
            ub = 1e6 * np.ones(selector_dim)
            return spaces.Box(-1 * ub, ub)

    @overrides
    def step(self, action):
        action = self.action_space.flatten(action)
        with self.low_policy.fix_selector(action):
            # print("The hier action is prefixed selector: {}".format(self.low_policy.pre_fix_selector))
            if isinstance(self.wrapped_env, FastMazeEnv):
                with self.wrapped_env.blank_maze():
                    frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg,
                                        reset_start_rollout=False, keep_rendered_rgbs=self.keep_rendered_rgb,
                                        animated=self.animate, speedup=1000)
                next_obs = self.wrapped_env.get_current_obs()
            elif isinstance(self.wrapped_env, NormalizedEnv) and isinstance(self.wrapped_env.wrapped_env, FastMazeEnv):
                with self.wrapped_env.wrapped_env.blank_maze():
                    frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg,
                                        reset_start_rollout=False, keep_rendered_rgbs=self.keep_rendered_rgb,
                                        animated=self.animate, speedup=1000)
                next_obs = self.wrapped_env.wrapped_env.get_current_obs()
            else:
                frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg,
                                    reset_start_rollout=False, keep_rendered_rgbs=self.keep_rendered_rgb,
                                    animated=self.animate, speedup=1000)
                next_obs = frac_path['observations'][-1]
            reward = np.sum(frac_path['rewards'])
            terminated = frac_path['terminated'][-1]
            done = self.time_steps_agg > len(
                frac_path['observations']) or terminated  # if the rollout was not maximal it was "done"
            # it would be better to add an extra flagg to this rollout to check if it was done in the last step
            last_agent_info = dict((k, val[-1]) for k, val in frac_path['agent_infos'].items())
            last_env_info = dict((k, val[-1]) for k, val in frac_path['env_infos'].items())
        # print("finished step of {}, with cummulated reward of: {}".format(len(frac_path['observations']), reward))
        if done:
            # if done I need to PAD the tensor so there is no mismatch. Pad with the last elem
            full_path = tensor_utils.pad_tensor_dict(frac_path, self.time_steps_agg, mode='last')
        else:
            full_path = frac_path

        return Step(next_obs, reward, done,
                    last_env_info=last_env_info, last_agent_info=last_agent_info, full_path=full_path)
        # the last kwargs will all go to env_info, so path['env_info']['full_path'] gives a dict with the full path!

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        # to use the visualization I need to append all paths
        expanded_paths = [tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path']) for path in paths]
        self.wrapped_env.log_diagnostics(expanded_paths, *args, **kwargs)

    def __str__(self):
        return "Hierarchized: %s" % self._wrapped_env


hierarchize_multi = HierarchizedMultiPoliEnv
