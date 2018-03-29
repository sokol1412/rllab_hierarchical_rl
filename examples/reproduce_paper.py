from rllab.algos.trpo import TRPO
from rllab.algos.tnpg import TNPG
from rllab.algos.ddpg import DDPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.swimmer3d_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
import time


def run_trpo_task(v):
    env = normalize(AntGatherEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)  # main text, section 5
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec) # suppl. section 2

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000, # suppl. section 2, Table 2
        max_path_length=500, # suppl. section 2, Table 2
        n_itr=200, # suppl. section 2, Table 2
        discount=0.99, # suppl. section 2, Table 2
        step_size=v["step_size"],
        #plot=True

    )
    algo.train()

#for step_size in [0.01, 0.05, 0.1]: #suppl. section 2, Table 4
#    for seed in [1, 11, 21, 31, 41]: #Figure 3. desciption

for step_size in [0.1]: #suppl. section 2, Table 4/////////chart (c)
    for seed in [1]: #Figure 3. desciption
        timestr = time.strftime("%Y%m%d-%H%M%S")
        run_experiment_lite(
            run_trpo_task,
            exp_name="_".join(["TRPO","step-",str(step_size),"seed-",str(seed),"time-",timestr]),
            exp_prefix="AntGatherer",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=seed,
            variant=dict(step_size=step_size, seed=seed),
        )


def run_TNPG_task(v):
    env = normalize(AntGatherEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)  # main text, section 5
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec) # suppl. section 2

    algo = TNPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000, # suppl. section 2, Table 2
        max_path_length=500, # suppl. section 2, Table 2
        n_itr=200, # suppl. section 2, Table 2
        discount=0.99, # suppl. section 2, Table 2
        step_size=v["step_size"],
        #plot=True

    )
    algo.train()

#for step_size in [0.01, 0.05, 0.1]: #suppl. section 2, Table 4
#    for seed in [1, 11, 21, 31, 41]: #Figure 3. desciption

for step_size in [0.5]: #suppl. section 2, Table 4/////////chart (c)
    for seed in [1]: #Figure 3. desciption
        timestr = time.strftime("%Y%m%d-%H%M%S")
        run_experiment_lite(
            run_TNPG_task,
            exp_name="_".join(["TNPG","step-",str(step_size),"seed-",str(seed),"time-",timestr]),
            exp_prefix="AntGatherer",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=seed,
            variant=dict(step_size=step_size, seed=seed),
        )


def run_DDPG_task(v):
    env = normalize(AntGatherEnv())

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers
        hidden_sizes=(400, 300)
    )

    es = OUStrategy(env_spec=env.spec)


    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=(400,300))
    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=64,
        max_path_length=500,
        min_pool_size=10000,
        epoch_length=1000,
        n_epochs=100, #here it should be around 3000 epochs, but takes QUITE long time...
        discount=0.99,
        scale_reward=0.1,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        #plot=True
    )
    algo.train()

#   for seed in [1, 11, 21, 31, 41]: #Figure 3. desciption
for seed in [1]: #Figure 3. desciption
    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_experiment_lite(
        run_DDPG_task,
        exp_name="_".join(["DDPG","step-",str("NOT_PROVIDED"),"seed-",str(seed),"time-",timestr]),
        exp_prefix="AntGatherer",
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        variant=dict(seed=seed),
        #plot=True #for DDPG algorithm, we need to set plot=True in both run_experiment_lite and DDPG constructor
    )
