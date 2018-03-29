import sys
import multiprocessing as mp
import numpy as np
import time
import os

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.misc import ext
from sandbox.ex2.parallel_trpo.sampler import WorkerBatchSampler
from sandbox.ex2.parallel_trpo.simple_container import SimpleContainer
from sandbox.ex2.utils.plot_utils import log_paths
import psutil

class ParallelBatchPolopt(RLAlgorithm):
    """
    Base class for parallelized batch sampling-based policy optimization methods.
    This includes various parallelized policy gradient methods like vpg, npg, ppo, trpo, etc.

    Here, parallelized is limited to mean: using multiprocessing package.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot_exemplar=False,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            n_parallel=1,
            set_cpu_affinity=False,
            cpu_assignments=None,
            serial_compile=True,
            clip_reward=False,
            exemplar_cls=None,
            exemplar_args=None,
            bonus_coeff=0,
            path_length_scheduler=None,
            log_memory_usage=True,
            avoid_duplicate_paths=False,
            path_replayer=None,
            tmax=-1,
            reset_freq=-1,
            eval_first=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.current_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.n_parallel = n_parallel
        self.set_cpu_affinity = set_cpu_affinity
        self.cpu_assignments = cpu_assignments
        self.serial_compile = serial_compile
        self.worker_batch_size = batch_size // n_parallel
        self.n_steps_collected = 0  # (set by sampler)
        self.avoid_duplicate_paths = avoid_duplicate_paths
        self.sampler = WorkerBatchSampler(self)
        self.clip_reward = clip_reward
        self.exemplar = None
        self.exemplar_cls = exemplar_cls
        self.exemplar_args = exemplar_args
        self.bonus_coeff = bonus_coeff
        self.path_length_scheduler = path_length_scheduler
        if path_length_scheduler is not None:
            self.path_length_scheduler.set_algo(self)
        self.log_memory_usage = log_memory_usage
        self.path_replayer = path_replayer
        self.tmax = tmax
        self.plot_exemplar = plot_exemplar
        self.reset_freq = reset_freq
        self.unpicklable_list = ["_par_objs","manager","shared_dict",
                                 "exemplar"]
        self.eval_first = eval_first

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        return {
            k: v for k, v in iter(self.__dict__.items())
            if k not in self.unpicklable_list
        }

    #
    # Serial methods.
    # (Either for calling before forking subprocesses, or subprocesses execute
    # it independently of each other.)
    #

    def _init_par_objs_batchpolopt(self):
        """
        Any init_par_objs() method in a derived class must call this method,
        and, following that, may append() the SimpleContainer objects as needed.
        """
        n = self.n_parallel
        self.rank = None
        shareds = SimpleContainer(
            sum_discounted_return=mp.RawArray('d', n),
            num_traj=mp.RawArray('i', n),
            sum_return=mp.RawArray('d', n),
            max_return=mp.RawArray('d', n),
            min_return=mp.RawArray('d', n),
            sum_raw_return=mp.RawArray('d', n),
            max_raw_return=mp.RawArray('d', n),
            min_raw_return=mp.RawArray('d', n),
            max_bonus=mp.RawArray('d', n),
            min_bonus=mp.RawArray('d', n),
            sum_bonus=mp.RawArray('d', n),
            sum_path_len=mp.RawArray('i',n),
            max_path_len=mp.RawArray('i',n),
            min_path_len=mp.RawArray('i',n),
            num_steps=mp.RawArray('i', n),
            num_valids=mp.RawArray('d', n),
            sum_ent=mp.RawArray('d', n),
        )
        ##HT: for explained variance (yeah I know it's clumsy)
        shareds.append(
            baseline_stats=SimpleContainer(
                y_sum_vec=mp.RawArray('d',n),
                y_square_sum_vec=mp.RawArray('d',n),
                y_pred_error_sum_vec=mp.RawArray('d',n),
                y_pred_error_square_sum_vec=mp.RawArray('d',n),
            )
        )
        barriers = SimpleContainer(
            dgnstc=mp.Barrier(n),
        )
        self._par_objs = (shareds, barriers)
        self.baseline.init_par_objs(n_parallel=n)
        if self.exemplar is not None:
            self.exemplar.init_par_objs(n_parallel=n)

    def init_par_objs(self):
        """
        Initialize all objects use for parallelism (called before forking).
        """
        raise NotImplementedError

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def prep_samples(self):
        """
        Used to prepare output from sampler.process_samples() for input to
        optimizer.optimize(), and used in force_compile().
        """
        raise NotImplementedError

    def force_compile(self, n_samples=100):
        """
        Serial - compile Theano (e.g. before spawning subprocesses, if desired)
        """
        logger.log("forcing Theano compilations...")
        paths = self.sampler.obtain_samples(n_samples)
        self.process_paths(paths)
        samples_data, _ = self.sampler.process_samples(paths)
        input_values = self.prep_samples(samples_data)
        self.optimizer.force_compile(input_values)
        self.baseline.force_compile()
        logger.log("all compiling complete")

    #
    # Main external method and its target for parallel subprocesses.
    #

    def train(self):
        self.init_opt()
        if self.serial_compile:
            self.force_compile()
        self.init_par_objs()
        self.manager = mp.Manager()
        self.shared_dict = self.manager.dict()

        if self.n_parallel == 1:
            self._train(0,self.shared_dict)
        else:
            processes = [mp.Process(target=self._train, args=(rank,self.shared_dict))
                for rank in range(self.n_parallel)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

    def process_paths(self, paths):
        if self.eval_first:
            for path in paths:
                path["raw_rewards"] = np.copy(path["rewards"])
                if self.clip_reward:
                    path["rewards"] = np.clip(path["raw_rewards"],-1,1)
                if self.exemplar is not None:
                    path["bonus_rewards"] = self.exemplar.predict(path)
            if self.exemplar is not None:
                if self.reset_freq > 0 and self.current_itr % self.reset_freq == 0:
                    self.exemplar.reset()
                    self.exemplar.fit(paths)
                self.exemplar.fit(paths)
        else:
            if self.exemplar is not None:
                if self.reset_freq > 0 and self.current_itr % self.reset_freq == 0:
                    self.exemplar.reset()
                    self.exemplar.fit(paths)
                self.exemplar.fit(paths)
            for path in paths:
                path["raw_rewards"] = np.copy(path["rewards"])
                if self.clip_reward:
                    path["rewards"] = np.clip(path["raw_rewards"],-1,1)
                if self.exemplar is not None:
                    path["bonus_rewards"] = self.exemplar.predict(path)

        if self.exemplar is not None:
            bonus_rewards = np.concatenate([path["bonus_rewards"].ravel() for path in paths])
            median_bonus = np.median(bonus_rewards)
            mean_discrim = np.mean(1 / (bonus_rewards + 1))
            for path in paths:
                path["bonus_rewards"] -= median_bonus
                path["rewards"] = path["rewards"] + self.bonus_coeff * path["bonus_rewards"]
            if self.rank == 0:
                logger.record_tabular('Median Bonus', median_bonus)
                logger.record_tabular('Mean Discrim', mean_discrim)

    def init_shared_dict(self,shared_dict):
        self.shared_dict = shared_dict
        if self.exemplar is not None:
            self.exemplar.init_shared_dict(shared_dict)

    def _train(self, rank, shared_dict):
        # Initialize separate exemplar per process
        if self.exemplar_cls is not None:
            self.exemplar = self.exemplar_cls(**self.exemplar_args)


        self.init_rank(rank)
        self.init_shared_dict(shared_dict)
        if self.rank == 0:
            start_time = time.time()

        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                self.update_algo_params(itr)
                if rank == 0:
                    logger.log("Collecting samples ...")
                paths = self.sampler.obtain_samples()
                if rank == 0:
                    logger.log("Processing paths...")
                self.process_paths(paths)

                if rank == 0:
                    logger.log("processing samples...")
                    if self.plot_exemplar:
                        if 'bonus_rewards' in paths[0]:
                            log_paths(paths[:20], 'traj_rewards', itr=itr)

                samples_data, dgnstc_data = self.sampler.process_samples(paths)

                self.log_diagnostics(itr, samples_data, dgnstc_data)  # (parallel)
                if rank == 0:
                    logger.log("optimizing policy...")

                if self.path_replayer is not None:
                    replayed_paths = self.path_replayer.replay_paths()
                    if len(replayed_paths) > 0:
                        self.process_paths(replayed_paths)
                        replayed_samples_data,_ = self.sampler.process_samples(replayed_paths)
                        samples_data = self.sampler.combine_samples([
                            samples_data, replayed_samples_data
                        ])
                    self.path_replayer.record_paths(paths)
                self.optimize_policy(itr, samples_data)  # (parallel)
                if rank == 0:
                    logger.log("fitting baseline...")
                # self.baseline.fit_by_samples_data(samples_data)  # (parallel)
                self.baseline.fit(paths)
                if rank == 0:
                    logger.log("fitted")
                    logger.log("saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)
                    params["algo"] = self
                    if self.store_paths:
                        # NOTE: Only paths from rank==0 worker will be saved.
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("saved")

                    logger.record_tabular("ElapsedTime",time.time()-start_time)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                      "continue...")
                if self.log_memory_usage:
                    process = psutil.Process(os.getpid())
                    print("Process %d memory usage: %.4f GB"%(rank,process.memory_info().rss / (1024**3)))
                    if self.rank == 0 and sys.platform == "linux":
                        print("Shared memory usage: %.4f GB"%(
                            process.memory_info().shared / (1024**3)
                        ))
                self.current_itr = itr + 1

    def update_algo_params(self,itr):
        if self.path_length_scheduler is not None:
            self.path_length_scheduler.update(itr)

    #
    # Parallelized methods and related.
    #

    def log_diagnostics(self, itr, samples_data, dgnstc_data):
            shareds, barriers = self._par_objs

            i = self.rank
            shareds.sum_discounted_return[i] = \
                np.sum([path["returns"][0] for path in samples_data["paths"]])
            undiscounted_returns = [sum(path["rewards"]) for path in samples_data["paths"]]
            undiscounted_raw_returns = [sum(path["raw_rewards"]) for path in samples_data["paths"]]
            shareds.num_traj[i] = len(undiscounted_returns)
            shareds.num_steps[i] = self.n_steps_collected
            # shareds.num_steps[i] = sum([len(path["rewards"]) for path in samples_data["paths"]])
            shareds.sum_return[i] = np.sum(undiscounted_returns)
            shareds.min_return[i] = np.min(undiscounted_returns)
            shareds.max_return[i] = np.max(undiscounted_returns)
            shareds.sum_raw_return[i] = np.sum(undiscounted_raw_returns)
            shareds.min_raw_return[i] = np.min(undiscounted_raw_returns)
            shareds.max_raw_return[i] = np.max(undiscounted_raw_returns)

            if self.exemplar is not None:
                # bonuses
                bonuses = np.concatenate([path["bonus_rewards"] for path in samples_data["paths"]])
                shareds.max_bonus[i] = np.max(bonuses)
                shareds.min_bonus[i] = np.min(bonuses)
                shareds.sum_bonus[i] = np.sum(bonuses)

            if not self.policy.recurrent:
                shareds.sum_ent[i] = np.sum(self.policy.distribution.entropy(
                    samples_data["agent_infos"]))
                shareds.num_valids[i] = 0
            else:
                raise NotImplementedError
                shareds.sum_ent[i] = np.sum(self.policy.distribution.entropy(
                    samples_data["agent_infos"]) * samples_data["valids"])
                shareds.num_valids[i] = np.sum(samples_data["valids"])

            # explained variance
            y_pred = np.concatenate(dgnstc_data["baselines"])
            y = np.concatenate(dgnstc_data["returns"])
            shareds.baseline_stats.y_sum_vec[i] = np.sum(y)
            shareds.baseline_stats.y_square_sum_vec[i] = np.sum(y**2)
            shareds.baseline_stats.y_pred_error_sum_vec[i] = np.sum(y-y_pred)
            shareds.baseline_stats.y_pred_error_square_sum_vec[i] = np.sum((y-y_pred)**2)

            # path lengths
            path_lens = [len(path["rewards"]) for path in samples_data["paths"]]
            shareds.sum_path_len[i] = np.sum(path_lens)
            shareds.max_path_len[i] = np.amax(path_lens)
            shareds.min_path_len[i] = np.amin(path_lens)

            barriers.dgnstc.wait()

            if self.rank == 0:
                self.env.log_diagnostics(samples_data["paths"])

                num_traj = sum(shareds.num_traj)
                n_steps = sum(shareds.num_steps)

                average_discounted_return = \
                    sum(shareds.sum_discounted_return) / num_traj

                if self.policy.recurrent:
                    ent = sum(shareds.sum_ent) / sum(shareds.num_valids)
                else:
                    ent = sum(shareds.sum_ent) / sum(shareds.num_steps)
                average_return = sum(shareds.sum_return) / num_traj
                max_return = max(shareds.max_return)
                min_return = min(shareds.min_return)

                average_raw_return = sum(shareds.sum_raw_return) / num_traj
                max_raw_return = max(shareds.max_raw_return)
                min_raw_return = min(shareds.min_raw_return)

                if self.exemplar is not None:
                    max_bonus = max(shareds.max_bonus)
                    min_bonus = min(shareds.min_bonus)
                    average_bonus = sum(shareds.sum_bonus) / n_steps

                # compute explained variance
                y_mean = sum(shareds.baseline_stats.y_sum_vec) / n_steps
                y_square_mean = sum(shareds.baseline_stats.y_square_sum_vec) / n_steps
                y_pred_error_mean = sum(shareds.baseline_stats.y_pred_error_sum_vec) / n_steps
                y_pred_error_square_mean = sum(shareds.baseline_stats.y_pred_error_square_sum_vec) / n_steps
                y_var = y_square_mean - y_mean**2
                y_pred_error_var = y_pred_error_square_mean - y_pred_error_mean**2
                if np.isclose(y_var,0):
                    ev = 0 # different from special.exaplained_variance_1d
                else:
                    ev = 1 - y_pred_error_var / (y_var + 1e-8)

                # path lens
                avg_path_len = sum(shareds.sum_path_len) / float(num_traj)
                max_path_len = max(shareds.max_path_len)
                min_path_len = min(shareds.min_path_len)

                logger.record_tabular('Iteration', itr)
                logger.record_tabular('ExplainedVariance', ev)
                logger.record_tabular('NumTrajs', num_traj)
                logger.record_tabular('NumSamples',n_steps)
                logger.record_tabular('Entropy', ent)
                logger.record_tabular('Perplexity', np.exp(ent))
                # logger.record_tabular('StdReturn', np.std(undiscounted_returns))
                logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
                logger.record_tabular('ReturnAverage', average_return)
                logger.record_tabular('ReturnMax', max_return)
                logger.record_tabular('ReturnMin', min_return)
                logger.record_tabular('RawReturnAverage', average_raw_return)
                logger.record_tabular('RawReturnMax', max_raw_return)
                logger.record_tabular('RawReturnMin', min_raw_return)
                logger.record_tabular('PathLenAverage',avg_path_len)
                logger.record_tabular('PathLenMax',max_path_len)
                logger.record_tabular('PathLenMin',min_path_len)
                if self.exemplar is not None:
                    logger.record_tabular('BonusRewardMax',max_bonus)
                    logger.record_tabular('BonusRewardMin',min_bonus)
                    logger.record_tabular('BonusRewardAverage',average_bonus)


        # NOTE: These others might only work if all path data is collected
        # centrally, could provide this as an option...might be easiest to build
        # multiprocessing pipes to send the data to the rank-0 process, so as
        # not to have to construct shared variables of specific sizes
        # beforehand.
        #
        # self.env.log_diagnostics(paths)
        # self.policy.log_diagnostics(paths)
        # self.baseline.log_diagnostics(paths)
        # self.exemplar.log_diagnostics(paths)

    def init_rank(self, rank):
        self.rank = rank
        if self.set_cpu_affinity:
            self._set_affinity(rank)
        self.baseline.init_rank(rank)
        self.optimizer.init_rank(rank)
        if self.exemplar is not None:
            self.exemplar.init_rank(rank)
        seed = ext.get_seed()
        if seed is None:
            # NOTE: Not sure if this is a good source for seed?
            seed = int(1e6 * np.random.rand())
        ext.set_seed(seed + rank)

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def _set_affinity(self, rank, verbose=False):
        """
        Check your logical cpu vs physical core configuration, use
        cpu_assignments list to put one worker per physical core.  Default
        behavior is to use logical cpus 0,1,2,...
        """
        import psutil
        if self.cpu_assignments is not None:
            n_assignments = len(self.cpu_assignments)
            assigned_affinity = [self.cpu_assignments[rank % n_assignments]]
        else:
            assigned_affinity = [rank % psutil.cpu_count()]
        p = psutil.Process()
        # NOTE: let psutil raise the error if invalid cpu assignment.
        try:
            p.cpu_affinity(assigned_affinity)
            if verbose:
                logger.log("\nRank: {},  CPU Affinity: {}".format(rank, p.cpu_affinity()))
        except AttributeError:
            logger.log("Cannot set CPU affinity (maybe in a Mac OS).")
