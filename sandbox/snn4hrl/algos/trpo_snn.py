from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.snn4hrl.algos.npo_snn_rewards import NPO_snn


class TRPO_snn(NPO_snn):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO_snn, self).__init__(optimizer=optimizer, **kwargs)
