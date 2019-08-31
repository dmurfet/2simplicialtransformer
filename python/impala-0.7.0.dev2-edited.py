from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy # MURFETD
import logging #MURFETD
import ray # MURFETD
from ray.exceptions import RayError # MURFETD
from ray.rllib.utils.schedules import ConstantSchedule #MURFETD
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID #MURFETD

from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
from ray.rllib.agents.impala.vtrace_policy_graph import VTracePolicyGraph
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.optimizers import AsyncSamplesOptimizer
from ray.rllib.optimizers.aso_tree_aggregator import TreeAggregator
from ray.rllib.utils.annotations import override
from ray.tune.trainable import Trainable
from ray.tune.trial import Resources

logger = logging.getLogger(__name__) #MURFETD

OPTIMIZER_SHARED_CONFIGS = [
    "lr",
    "num_envs_per_worker",
    "num_gpus",
    "sample_batch_size",
    "train_batch_size",
    "replay_buffer_num_slots",
    "replay_proportion",
    "num_data_loader_buffers",
    "max_sample_requests_in_flight_per_worker",
    "broadcast_interval",
    "num_sgd_iter",
    "minibatch_buffer_size",
    "num_aggregation_workers",
]

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # V-trace params (see vtrace.py).
    "vtrace": True,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,

    # System params.
    #
    # == Overview of data flow in IMPALA ==
    # 1. Policy evaluation in parallel across `num_workers` actors produces
    #    batches of size `sample_batch_size * num_envs_per_worker`.
    # 2. If enabled, the replay buffer stores and produces batches of size
    #    `sample_batch_size * num_envs_per_worker`.
    # 3. If enabled, the minibatch ring buffer stores and replays batches of
    #    size `train_batch_size` up to `num_sgd_iter` times per batch.
    # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
    #    on batches of size `train_batch_size`.
    #
    "sample_batch_size": 50,
    "train_batch_size": 500,
    "min_iter_time_s": 10,
    "num_workers": 2,
    # number of GPUs the learner should use.
    "num_gpus": 1,
    # set >1 to load data into GPUs in parallel. Increases GPU memory usage
    # proportionally with the number of buffers.
    "num_data_loader_buffers": 1,
    # how many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    "minibatch_buffer_size": 1,
    # number of passes to make over each train batch
    "num_sgd_iter": 1,
    # set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    "replay_proportion": 0.0,
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * sample_batch_size).
    "replay_buffer_num_slots": 0,
    # max queue size for train batches feeding into the learner
    "learner_queue_size": 16,
    # level of queuing for sampling.
    "max_sample_requests_in_flight_per_worker": 2,
    # max number of workers to broadcast one set of weights to
    "broadcast_interval": 1,
    # use intermediate actors for multi-level aggregation. This can make sense
    # if ingesting >2GB/s of samples, or if the data requires decompression.
    "num_aggregation_workers": 0,

    # Learning params.
    "grad_clip": 40.0,
    # either "adam" or "rmsprop"
    "opt_type": "adam",
    "lr": 0.0005,
    "lr_schedule": None,
    # rmsprop considered
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.1,
    # balancing the three losses
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,

    # use fake (infinite speed) sampler for testing
    "_fake_sampler": False,
})
# __sphinx_doc_end__
# yapf: enable


class ImpalaTrainer(Trainer):
    """IMPALA implementation using DeepMind's V-trace."""

    _name = "IMPALA"
    _default_config = DEFAULT_CONFIG
    _policy_graph = VTracePolicyGraph

    @override(Trainer)
    def _init(self, config, env_creator):
        for k in OPTIMIZER_SHARED_CONFIGS:
            if k not in config["optimizer"]:
                config["optimizer"][k] = config[k]
        policy_cls = self._get_policy_graph()
        self.local_evaluator = self.make_local_evaluator(
            self.env_creator, policy_cls)

        if self.config["num_aggregation_workers"] > 0:
            # Create co-located aggregator actors first for placement pref
            aggregators = TreeAggregator.precreate_aggregators(
                self.config["num_aggregation_workers"])

        self.remote_evaluators = self.make_remote_evaluators(
            env_creator, policy_cls, config["num_workers"])
        self.optimizer = AsyncSamplesOptimizer(self.local_evaluator,
                                               self.remote_evaluators,
                                               **config["optimizer"])
        if config["entropy_coeff"] < 0:
            raise DeprecationWarning("entropy_coeff must be >= 0")

        if self.config["num_aggregation_workers"] > 0:
            # Assign the pre-created aggregators to the optimizer
            self.optimizer.aggregator.init(aggregators)

    @classmethod
    @override(Trainable)
    def default_resource_request(cls, config):
        cf = dict(cls._default_config, **config)
        Trainer._validate_config(cf)
        return Resources(
            cpu=cf["num_cpus_for_driver"],
            gpu=cf["num_gpus"],
            extra_cpu=cf["num_cpus_per_worker"] * cf["num_workers"] +
            cf["num_aggregation_workers"],
            extra_gpu=cf["num_gpus_per_worker"] * cf["num_workers"])

    @override(Trainer)
    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        start = time.time()
        self.optimizer.step()
        while (time.time() - start < self.config["min_iter_time_s"]
               or self.optimizer.num_steps_sampled == prev_steps):
            self.optimizer.step()
        result = self.collect_metrics()
        result.update(timesteps_this_iter=self.optimizer.num_steps_sampled -
                      prev_steps)
        return result

    def _get_policy_graph(self):
        if self.config["vtrace"]:
            policy_cls = self._policy_graph
        else:
            policy_cls = A3CPolicyGraph
        return policy_cls

    # MURFETD
    def reset_config(self, new_config):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(new_config)
        self.config = config
        
        # see LearningRateSchedule.__init__(self, self.config["lr"],self.config["lr_schedule"])
        # in vtrace_policy_graph.py
        # see policy_evaluator.py
       
        ev = self.optimizer.local_evaluator
        p = ev.policy_map[DEFAULT_POLICY_ID]
        p.lr_schedule = ConstantSchedule(self.config["lr"])
        p.cur_lr.load(self.config["lr"],session=ev.tf_sess)
        
        return True

    @override(Trainer)
    def _try_recover(self):
        """Try to identify and blacklist any unhealthy workers.

        This method is called after an unexpected remote error is encountered
        from a worker. It issues check requests to all current workers and
        blacklists any that respond with error. If no healthy workers remain,
        an error is raised.
    
        MURFETD: some changes from Ray-0.7.0-dev2
        """

        if not self._has_policy_optimizer():
            raise NotImplementedError(
                "Recovery is not supported for this algorithm")

        logger.info("Health checking all workers...")
        checks = []
        for ev in self.optimizer.remote_evaluators:
            _, obj_id = ev.sample_with_count.remote()
            checks.append(obj_id)

        healthy_evaluators = []
        for i, obj_id in enumerate(checks):
            ev = self.optimizer.remote_evaluators[i]
            try:
                ray.get(obj_id)
                healthy_evaluators.append(ev)
                logger.info("Worker {} looks healthy".format(i + 1))
            except RayError:
                logger.exception("Blacklisting worker {}".format(i + 1))
                try:
                    ev.__ray_terminate__.remote()
                except Exception:
                    logger.exception("Error terminating unhealthy worker")

        if len(healthy_evaluators) < 1:
            raise RuntimeError(
                "Not enough healthy workers remain to continue.")

        # MURFETD (add additional new remote_evaluators)
        #num_new_evaluators = len(checks) - len(healthy_evaluators)
    
        #new_evaluators = self.make_remote_evaluators(
        #        self.env_creator, self._get_policy_graph(), num_new_evaluators)
            
        #healthy_evaluators.extend(new_evaluators)
    
        # MURFETD (keep our remote_evaluator list in sync with the optimizer/aggregator)
        #self.remote_evaluators = healthy_evaluators
    
        self.optimizer.reset(healthy_evaluators)
