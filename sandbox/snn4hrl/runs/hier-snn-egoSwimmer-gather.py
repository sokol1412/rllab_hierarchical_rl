"""
SwimmerGather: find good size to compare agains baseline
"""

# imports -----------------------------------------------------
import math

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.config_personal import *
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.snn4hrl.algos.trpo_snn import TRPO_snn
from sandbox.snn4hrl.envs.hierarchized_snn_env import hierarchize_snn
from sandbox.snn4hrl.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from sandbox.snn4hrl.policies.categorical_mlp_policy import CategoricalMLPPolicy

stub(globals())

# exp setup --------------------------------------------------------
mode = "local"
n_parallel = 4

exp_dir = '/home/sokol/Pulpit/Magisterka_Wladek/Magisterka/rllab_wadim_fork/data/local/egoSwimmer-snn/'
for dir in os.listdir(exp_dir):
    if 'Figure' not in dir and os.path.isfile(os.path.join(exp_dir, dir, 'params.pkl')):
        pkl_path = os.path.join(exp_dir, dir, 'params.pkl')
        print("hier for : ", pkl_path)

        for time_step_agg in [10, 50, 100]:

            for activity_range in [6, 10, 15]:
                inner_env = normalize(SwimmerGatherEnv(activity_range=activity_range, sensor_range=activity_range,
                                                       sensor_span=math.pi * 2, ego_obs=True))
                env = hierarchize_snn(inner_env, time_steps_agg=time_step_agg, pkl_path=pkl_path,
                                      # animate=True,
                                      )

                policy = CategoricalMLPPolicy(
                    env_spec=env.spec,
                )

                baseline = LinearFeatureBaseline(env_spec=env.spec)

                # bonus_evaluators = [GridBonusEvaluator(mesh_density=mesh_density, visitation_bonus=1, snn_H_bonus=0)]
                # reward_coef_bonus = [reward_coef]

                algo = TRPO_snn(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    self_normalize=True,
                    log_deterministic=True,
                    # reward_coef=reward_coef,
                    # bonus_evaluator=bonus_evaluators,
                    # reward_coef_bonus=reward_coef_bonus,
                    batch_size=5e5 / time_step_agg,
                    whole_paths=True,
                    max_path_length=5e3 / time_step_agg * activity_range / 6.,  # correct for larger envs
                    n_itr=2000,
                    discount=0.99,
                    step_size=0.01,
                )

                for s in [0]:  # range(10, 110, 10):  # [10, 20, 30, 40, 50]:
                    exp_prefix = 'hier-snn-egoSwimmer-gather'
                    exp_name = exp_prefix + '{}range_{}agg_{}pl_PRE{}_{}'.format(activity_range,
                                                                                 time_step_agg, int(
                            5e3 / time_step_agg * activity_range / 6.),
                                                                                 pkl_path.split('/')[-2], s)

                    run_experiment_lite(
                        stub_method_call=algo.train(),
                        mode=mode,
                        use_cloudpickle=False,
                        pre_commands=['pip install --upgrade pip',
                                      'pip install --upgrade theano',
                                      ],
                        # Number of parallel workers for sampling
                        n_parallel=n_parallel,
                        # Only keep the snapshot parameters for the last iteration
                        snapshot_mode="last",
                        seed=s,
                        # Save to data/local/exp_prefix/exp_name/
                        exp_prefix=exp_prefix,
                        exp_name=exp_name,
                    )
