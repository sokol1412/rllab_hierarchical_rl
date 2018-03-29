import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano
import theano.tensor as TT
import numpy as np
from contextlib import contextmanager

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.sampler.utils import rollout  # I need this for logging the diagnostics: run the policy with all diff latents

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc import autoargs
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.distributions.bernoulli import Bernoulli
# from rllab.distributions.categorical import Categorical

from sandbox.snn4hrl.distributions.categorical import Categorical_oneAxis as Categorical


class GaussianMLPPolicy_snn(StochasticPolicy, LasagnePowered, Serializable):
    """
    This stochastic policy allows to pick the latent distribution (Categorical in the paper), its dimension and
    its integration with the observations.
    """
    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('std_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected layers for std, note'
                       'there is a difference in semantics than above: here an empty'
                       'list means that std is independent of input and the last size is ignored')
    @autoargs.arg('initial_std', type=float,
                  help='Initial std')
    @autoargs.arg('std_trainable', type=bool,
                  help='Is std trainable')
    @autoargs.arg('output_nl', type=str,
                  help='nonlinearity for the output layer')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    @autoargs.arg('bn', type=bool,
                  help='whether to apply batch normalization to hidden layers')
    def __init__(
            self,
            env_spec,
            latent_dim=2,
            latent_name='bernoulli',
            bilinear_integration=False,
            resample=False,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            std_hidden_nonlinearity=NL.tanh,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            min_std=1e-4,
    ):
        """
        :param latent_dim: dimension of the latent variables
        :param latent_name: distribution of the latent variables
        :param bilinear_integration: Boolean indicator of bilinear integration or simple concatenation
        :param resample: Boolean indicator of resampling at every step or only at the start of the rollout (or whenever
        agent is reset, which can happen several times along the rollout with rollout in utils_snn)
        """
        self.latent_dim = latent_dim  ##could I avoid needing this self for the get_action?
        self.latent_name = latent_name
        self.bilinear_integration = bilinear_integration
        self.resample = resample
        self.min_std = min_std
        self.hidden_sizes = hidden_sizes

        self.pre_fix_latent = np.array([])  # if this is not empty when using reset() it will use this latent
        self.latent_fix = np.array([])  # this will hold the latents variable sampled in reset()
        self._set_std_to_0 = False

        if latent_name == 'normal':
            self.latent_dist = DiagonalGaussian(self.latent_dim)
            self.latent_dist_info = dict(mean=np.zeros(self.latent_dim), log_std=np.zeros(self.latent_dim))
        elif latent_name == 'bernoulli':
            self.latent_dist = Bernoulli(self.latent_dim)
            self.latent_dist_info = dict(p=0.5 * np.ones(self.latent_dim))
        elif latent_name == 'categorical':
            self.latent_dist = Categorical(self.latent_dim)
            if self.latent_dim > 0:
                self.latent_dist_info = dict(prob=1./self.latent_dim * np.ones(self.latent_dim))
            else:
                self.latent_dist_info = dict(prob=np.ones(self.latent_dim))
        else:
            raise NotImplementedError

        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        if self.bilinear_integration:
            obs_dim = env_spec.observation_space.flat_dim + latent_dim +\
                      env_spec.observation_space.flat_dim * latent_dim
        else:
            obs_dim = env_spec.observation_space.flat_dim + latent_dim  # here only if concat.

        action_dim = env_spec.action_space.flat_dim

        mean_network = MLP(
            input_shape=(obs_dim,),
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            name="meanMLP",
        )

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_layer.input_var

        if adaptive_std:
            l_log_std = MLP(
                input_shape=(obs_dim,),
                input_var=obs_var,
                output_dim=action_dim,
                hidden_sizes=std_hidden_sizes,
                hidden_nonlinearity=std_hidden_nonlinearity,
                output_nonlinearity=None,
                name="log_stdMLP"
            ).output_layer
        else:
            l_log_std = ParamLayer(
                mean_network.input_layer,
                num_units=action_dim,
                param=lasagne.init.Constant(np.log(init_std)),
                name="output_log_std",
                trainable=learn_std,
            )

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))

        self._l_mean = l_mean
        self._l_log_std = l_log_std

        self._dist = DiagonalGaussian(action_dim)

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(GaussianMLPPolicy_snn, self).__init__(env_spec)

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

# #  this is currently not used, although it could, in dist_info_sym and in get_actions. Also we could refactor all..
#         # this would actually be WRONG with the current obs_var definition
#         latent_var = Box(low=-np.inf, high=np.inf, shape=(1,)).new_tensor_variable('latents', extra_dims=1)
#
#         extended_obs_var = TT.concatenate([obs_var, latent_var,
#                                            TT.flatten(obs_var[:, :, np.newaxis] * latent_var[:, np.newaxis, :],
#                                                       outdim=2)]
#                                           , axis=1)
#         self._extended_obs_var = ext.compile_function(
#             inputs=[obs_var, latent_var],
#             outputs=[extended_obs_var]
#         )

    @property
    def latent_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(1,))

    # the mean and var now also depend on the particular latents sampled
    def dist_info_sym(self, obs_var, latent_var=None):  # this is ment to be for one path!
        # now this is not doing anything! And for computing the dist_info_vars of npo_snn_rewardMI it doesn't work
        if latent_var is None:
            latent_var1 = theano.shared(np.expand_dims(self.latent_fix, axis=0))  # new fix to avoid putting the latent as an input: just take the one fixed!
            latent_var = TT.tile(latent_var1, [obs_var.shape[0], 1])

        # generate the generalized input (append latents to obs.)
        if self.bilinear_integration:
            extended_obs_var = TT.concatenate([obs_var, latent_var,
                                               TT.flatten(obs_var[:, :, np.newaxis] * latent_var[:, np.newaxis, :],
                                                          outdim=2)]
                                              , axis=1)
        else:
            extended_obs_var = TT.concatenate([obs_var, latent_var], axis=1)
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], extended_obs_var)
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        observations = np.array(observations)  # needed to do the outer product for the bilinear
        if self.latent_dim:
            if self.resample:
                latents = [self.latent_dist.sample(self.latent_dist_info) for _ in observations]
                print('resampling the latents')
            else:
                if not np.size(self.latent_fix) == self.latent_dim:  # we decide to reset based on if smthing in the fix
                    self.reset()
                if len(self.pre_fix_latent) == self.latent_dim:  # If we have a pre_fix, reset will put the latent to it
                    self.reset()  # this overwrites the latent sampled or in latent_fix
                latents = np.tile(self.latent_fix, [len(observations), 1])  # maybe a broadcast operation better...
            if self.bilinear_integration:
                extended_obs = np.concatenate([observations, latents,
                                               np.reshape(
                                                   observations[:, :, np.newaxis] * latents[:, np.newaxis, :],
                                                   (observations.shape[0], -1))],
                                              axis=1)
            else:
                extended_obs = np.concatenate([observations, latents], axis=1)
        else:
            latents = np.array([[]] * len(observations))
            extended_obs = observations
        # make mean, log_std also depend on the latents (as observ.)
        mean, log_std = self._f_dist(extended_obs)

        if self._set_std_to_0:
            actions = mean
            log_std = -1e6 * np.ones_like(log_std)
        else:
            rnd = np.random.normal(size=mean.shape)
            actions = rnd * np.exp(log_std) + mean
        return actions, dict(mean=mean, log_std=log_std, latents=latents)

    def set_pre_fix_latent(self, latent):
        self.pre_fix_latent = np.array(latent)

    def unset_pre_fix_latent(self):
        self.pre_fix_latent = np.array([])

    @contextmanager
    def fix_latent(self, latent):
        self.pre_fix_latent = np.array(latent)
        yield
        self.pre_fix_latent = np.array([])

    @contextmanager
    def set_std_to_0(self):
        self._set_std_to_0 = True
        yield
        self._set_std_to_0 = False

    @overrides
    def reset(self, force_resample_lat=False):  # executed at the start of every rollout. Will fix the latent if needed.
        if not self.resample and self.latent_dim:
            if self.pre_fix_latent.size > 0 and not force_resample_lat:
                self.latent_fix = self.pre_fix_latent
            else:
                self.latent_fix = self.latent_dist.sample(self.latent_dist_info)
        else:
            pass

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('MaxPolicyStd', np.max(np.exp(log_stds)))
        logger.record_tabular('MinPolicyStd', np.min(np.exp(log_stds)))
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist

    def log_likelihood(self, actions, agent_infos, action_only=True):
        # First compute logli of the action. This assumes the latents FIX to whatever was sampled, and hence we only
        # need to use the mean and log_std, but not any information about the latents
        logli = self._dist.log_likelihood(actions, agent_infos)
        if not action_only:
            raise NotImplementedError
            #   if not action_only:
            #       for idx, latent_name in enumerate(self._latent_distributions):
            #           latent_var = dist_info["latent_%d" % idx]
            #           prefix = "latent_%d_" % idx
            #           latent_dist_info = {k[len(prefix):]: v for k, v in dist_info.iteritems() if k.startswith(
            #               prefix)}
            #           logli += latent_name.log_likelihood(latent_var, latent_dist_info)
        return logli
