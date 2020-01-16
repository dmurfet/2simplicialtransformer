# Ray structural overview

Some survival tips: use log files and record experiment_id's from RLlib, so you can find the checkpoint files. Also save the checkpoint files that look like `experiment_state-2019-08-01_12-14-08.json` as you will need them to resume trials later.

## High level concepts

First read [this](https://ray.readthedocs.io/en/latest/tutorial.html) for a general overview of Ray and its terminology (get, put, remote functions, actors being the keywords). From the documentation:

> Ray is a distributed execution engine. The same code can be run on a single machine to achieve efficient
> multiprocessing, and it can be used on a cluster for large computations.
> When using Ray, several processes are involved.
>    - Multiple **worker** processes execute tasks and store results in object stores. Each worker is a separate process.
>    - One **object store** per node stores immutable objects in shared memory and allows workers to efficiently share objects on the same node with minimal copying and deserialization.
>    - One **raylet** per node assigns tasks to workers on the same node.
>    - A **driver** is the Python process that the user controls. For example, if the user is running a script or using a Python shell, then the driver is the Python process that runs the script or the shell. A driver is similar to a worker in that it can submit tasks to its raylet and get objects from the object store, but it is different in that the raylet will not assign tasks to the driver to be executed.
>    - A **Redis server** maintains much of the system’s state. For example, it keeps track of which objects live on which machines and of the task specifications (but not data). It can also be queried directly for debugging purposes.

The code that is executed on the cluster in our case is IMPALA training and rollouts, using RLlib, and this code is wrapped up as class methods for various classes (e.g. ImpalaTrainer). So the way that we actually execute code on the cluster is through *Ray actors*, see [this](https://ray.readthedocs.io/en/latest/actors.html):

> Ray extends the dataflow model with actors. An actor is essentially a stateful worker (or a service). When a new actor is instantiated, a new worker is created, and methods of the actor are scheduled on that specific worker and can access and mutate the state of that worker.

Here `worker` is used in several slightly different ways, but what it means is that when a new actor is instantiated (that is, an instance of the class is created) a new thread is created on a particular machine in the cluster, and that thread is then associated to that particular class instance, in the sense that methods of that instance when executed with `.remote()` will be executed *in that thread on that machine*. This is for instance how the IMPALA workers are distributed across the cluster, by creating instances of the class `PolicyEvaluator` that are actors in this sense (i.e. they are class instances tagged with `@ray.remote` so that they actually execute their methods on a machine in the cluster).

So what classes are involved? The main ones are [Trial Runner](https://github.com/ray-project/ray/blob/master/python/ray/tune/trial_runner.py), [Trial Executor](https://github.com/ray-project/ray/blob/master/python/ray/tune/ray_trial_executor.py), [Trainer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py), [Trainable](https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable.py), [ImpalaTrainer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/impala/impala.py) and the [PBT scheduler](https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pbt.py). First of all, here is a useful class hierarchy: `ImpalaTrainer < Trainer < Trainable` where `<` means "is a subclass of".

## The core loop

Let begin with the **core loop** that controls everything else, which is `ray.tune` [here](https://github.com/ray-project/ray/blob/master/python/ray/tune/tune.py). With `scheduler = pbt` we have

```
def run(...)

experiment = Experiment("IMPALA", ...., checkpoint_freq, ...)
scheduler = scheduler or FIFOScheduler()
search_alg = search_alg or BasicVariantGenerator()
trial_executor = None
search_alg.add_configurations([experiment])

runner = TrialRunner(
            search_alg,
            scheduler=scheduler,
            queue_trials=queue_trials,
            reuse_actors=reuse_actors,
            trial_executor=trial_executor)
            
while not runner.is_finished():
    runner.step()

return runner.get_trials()
```
So the first level of organisation is a **TrialRunner** which is responsible for scheduling **Trials** and the main loop executes `step` on the TrialRunner until it is finished. There is only one TrialRunner but there are many Trials (each one representing one PBT member). The way to understand this is that the SearchAlgorithm is the gadget responsible for feeding trials to the TrialRunner. So the first thing to understand is TrialRunner:

```
class TrialRunner(object):
    """A TrialRunner implements the event loop for scheduling trials on Ray.

    The main job of TrialRunner is scheduling trials to efficiently use cluster
    resources, without overloading the cluster.
    While Ray itself provides resource management for tasks and actors, this is
    not sufficient when scheduling trials that may instantiate multiple actors.
    This is because if insufficient resources are available, concurrent trials
    could deadlock waiting for new resources to become available. Furthermore,
    oversubscribing the cluster could degrade training performance, leading to
    misleading benchmark results.
    """
        
def step(self):
        """Runs one step of the trial event loop.
        Callers should typically run this method repeatedly in a loop. They
        may inspect or modify the runner's state in between calls to step().
        """
        next_trial = self._get_next_trial()  # blocking
        if next_trial is not None:
            self.trial_executor.start_trial(next_trial)
        elif self.trial_executor.get_running_trials():
            self._process_events()  # blocking
```

Note that the TrialRunner does not itself actually start trials, or stop them, it uses the TrialExecutor to do this, which in our case is an instance of [RayTrialExecutor](https://github.com/ray-project/ray/blob/master/python/ray/tune/ray_trial_executor.py) (see `__init__`). Meanwhile the scheduler is used to make the decisions about which trials to run, when to pause them, etc. In our case this is PBT. 

### Trial runner: starting trials

Here when we start a Trial it means *for one iteration*. If you look at `_get_next_trial` you'll see that it asks [PBT](https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pbt.py) `choose_trial_to_run`. This function looks at the list of PENDING and PAUSED trials which can be accommodated on the cluster and returns one to run `start_trial` on. If there are no such trials (for instance the Trial Runner in previous steps has successfuly started as many trials as we can handle) then `_get_next_trial` returns None and we move on to the `_process_events` part of the loop. This is where the main logic of training happens. But before describing that, it is important to understand what happens in `start_trial` from RayTrialExecutor:

```
def _start_trial(self, trial, checkpoint=None):
        """Starts trial and restores last result if trial was paused.
        """
        prior_status = trial.status
        self.set_status(trial, Trial.RUNNING)
        trial.runner = self._setup_runner(
            trial,
            reuse_allowed=checkpoint is not None
            or trial._checkpoint.value is not None)
        if not self.restore(trial, checkpoint):
            if trial.status == Trial.ERROR:
                raise RuntimeError(
                    "Restore from checkpoint failed for Trial {}.".format(
                        str(trial)))

        previous_run = self._find_item(self._paused, trial)
        if (prior_status == Trial.PAUSED and previous_run):
            # If Trial was in flight when paused, self._paused stores result.
            self._paused.pop(previous_run[0])
            self._running[previous_run[0]] = trial
        else:
            self._train(trial)
```
This checkpoint is discussed later. In any case, this is how we finally get to `_train` (again in RayTrialExecutor):

```
    def _train(self, trial):
        """Start one iteration of training and save remote id."""

        assert trial.status == Trial.RUNNING, trial.status
        remote = trial.runner.train.remote()

        self._running[remote] = trial
```
The `trial.runner` is a Ray actor which is an instance of ImpalaTrainer (**not** an instance of TrialRunner, jesus). So we run `train` remotely. Note that only `_train` is defined in ImpalaTrainer, so this falls through to `train` in [Trainer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py) which wraps `train` in [Trainable](https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable.py) which in turn calls `_train` which is where the IMPALA specific code is executed. So the above code block submits the task `Trainer.train()` to a raylet and returns an objectID `remote` which is used as a key in the `self._running` dictionary. Later we will use `ray.get` on this objectID to get the actual return value, which is the result of the training (meaning, one iteration of IMPALA). So once `train` has been executed remotely on the ImpalaTrainer instance associated to each Trial, the `step` function of the Trial Runner will move on to processing these results

### Trial runner: processing results

The function `_process_events` asks the trial executor for the next available trial and then runs `self._process_trial(trial)` on it (to be clear, this is a function in [Trial Runner](https://github.com/ray-project/ray/blob/master/python/ray/tune/trial_runner.py)). And in `_process_trial` we find

```
def _process_trial(self, trial):
        try:
            result = self.trial_executor.fetch_result(trial)            
            ...
            self._checkpoint_trial_if_needed(trial)
            ...
            decision = self._scheduler_alg.on_trial_result(self, trial, result)
            ...
            if decision == TrialScheduler.CONTINUE:
                self.trial_executor.continue_training(trial)
        except Exception:
            logger.exception("Error processing event.")
            error_msg = traceback.format_exc()
            if trial.status == Trial.RUNNING:
                if trial.should_recover():
                    self._try_recover(trial, error_msg)
                else:
                    self._scheduler_alg.on_trial_error(self, trial)
                    self._search_alg.on_trial_complete(
                        trial.trial_id, error=True)
                    self.trial_executor.stop_trial(
                        trial, error=True, error_msg=error_msg)
    
```
We will discuss `fetch_result` in a moment. But generically speaking, `decision = TrialScheduler.CONTINUE` and `continue_training` just runs `self._train(trial)` again, as discussed above. This continues until there is a resource issue so that some Trials must be transitioned to PENDING (I do not know how this works) or the experiment concludes. Note that calling `_train` is non-blocking, as it just executes a remote function call on the ImpalaTrainer and returns an objectID.

Anyway, the core logic of processing results is contained in `trial_executor.fetch_result()` which we find defined in [RayTrialExecutor](https://github.com/ray-project/ray/blob/master/python/ray/tune/ray_trial_executor.py):
```
    def fetch_result(self, trial):
        """Fetches one result of the running trials.
        Returns:
            Result of the most recent trial training run."""
        trial_future = self._find_item(self._running, trial)
        self._running.pop(trial_future[0])
        
        result = ray.get(trial_future[0])
```
So what the hell is this: the core logic is hidden behind a `ray.get` call to some opaque thing called a trial future. *Sigh*. But it's not so bad, if you read the above. Recall that `self._running` is a dictionary which has trials as values and as keys object IDs that are the IDs in the object store of return values of `train()` called on instances of ImpalaTrainer. This business with `_find_item` is doing the following: take the dictionary of running trials, form the list of keys whose value is `trial` and take the first entry in that list. That is `trial_future[0]`. This is pretty stupid because the Python documentation says that the order of items in `dictionary.items()` is not guaranteed to be the same across runs, but whatever. I think that in our case `trial_future` is always a list with one element anyway, so `trial_future[0]` is just the objectID of the return value of `train()` on our ImpalaTrainer. We then pop this from the `_running` dictionary and retrieve it from the object store.

**SUMMARY:** The Trial Runner takes PENDING and PAUSED trials, fills up the available resources, remotely runs `train()` on all our ImpalaTrainer instances, and then retrieves the results returned form these functions from the object store using `ray.get`. These return values are then processed, and `train()` is run again until the Trial is completed or resources are no longer sufficient to keep the Trial in a RUNNING state.

### What is a Trial?

```
class Trial(object):
    """A trial object holds the state for one model training run.
    Trials are themselves managed by the TrialRunner class, which implements
    the event loop for submitting trial runs to a Ray cluster.
    Trials start in the PENDING state, and transition to RUNNING once started.
    On error it transitions to ERROR, otherwise TERMINATED on success.
    """
```
Where do Trials come from? As we have already explained, in `tune.run` the SearchAlgorithm takes the experiments and feeds them to the TrialRunner. So if Trials are Experiments, how does the ImpalaTrainer get into an [Experiment](https://github.com/ray-project/ray/blob/master/python/ray/tune/experiment.py)?

```
class Experiment(object):
    """Tracks experiment specifications.
    Implicitly registers the Trainable if needed.
```

Note that `should_recover()` in `_process_trial` of TrialRunner is just `return (self.checkpoint_freq > 0 and (self.num_failures < self.max_failures or self.max_failures < 0))`.

## Checkpoints

Let us now enumerate the various kinds of checkpoints that show up in the above. 

* Every step the trial runner runs `self.checkpoint()` which amounts to saving *to disk* a JSON file containing

```
runner_state = {
            "checkpoints": list(
                self.trial_executor.get_checkpoints().values()),
            "runner_data": self.__getstate__(),
            "timestamp": time.time()
        }
```

These checkpoints are not cumulative, that is, they overwrite the checkpoint from the last step. Note that `get_checkpoints` is actually defined in the base [TrialExecutor](https://github.com/ray-project/ray/blob/master/python/ray/tune/trial_executor.py) and returns a copy of the instance variable `_cached_trial_state` which is a dictionary mapping the trial ID to pickled metadata (I am not clear on how this variable actually gets the cached trial state, however) but it appears to come from running `trial.__getstate__()`.

* Every time the Trial runner runs `_process_trial` on a trial it asks the Trial if it should be checkpointed (according to the global configuration flag of the experiment `checkpoint_freq`) and if yes, then it saves a checkpoint *to disk* with `self.trial_executor.save(trial, storage=Checkpoint.DISK)`. This updates `trial._checkpoint.storage = storage` and `trial._checkpoint.last_result = trial.last_result` and this is the same checkpoint that is later used in RayTrialExecutor `start_trial` where in `self.restore(trial, checkpoint)`.

* PBT (see below) uses a separate and parallel set of checkpoints *in memory* which are created with `trial_runner.trial_executor.save(trial, Checkpoint.MEMORY)`. These are the checkpoints that PBT uses in its exploit phase.

# IMPALA

Each trial is an instance of [ImpalaTrainer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/impala/impala.py) which conceptually is made up of a central IMPALA learner and IMPALA actors which generate experience that comes back as sample batches to the learner. Note that both the learner and the actors need the weights of the policy model, the learner for backpropagation and gradient descent, and the actors for inference (i.e. rollouts). In this section we examine the details of how IMPALA interacts with the Ray distributed computation model. In particular, how *IMPALA actors* map to *Ray actors* (which are instances of classes that get allocated to worker nodes on the cluster, such that methods are executed on those instances with `method.remote()`).

As we will explain, for each member of the PBT population a Ray Actor of type ImpalaTrainer is created and allocated to some worker node. This Ray Actor in turn will spawn one `local_evaluator` and multiple `remote_evaluators` which (as far as I can tell) could be on yet another machine in the cluster.

## ImpalaTrainer as a Ray actor

How do Ray actors arise in IMPALA training on RLlib? To answer this question requires chasing more rabbits down more holes. We need to understand *how Trials get into the Trial Runner*. First of all in TrialRunner `_get_next_trial` we run into `_update_trial_queue` which runs

```
trials = self._search_alg.next_trials()
for trial in trials:
    self.add_trial(trial)
```
The search algorithm is by default is [BasicVariantGenerator](https://github.com/ray-project/ray/blob/master/python/ray/tune/suggest/basic_variant.py) the function `next_trials()` of which returns a shuffled version of `self._trial_generator`. But how does anything get into this variable? Now we recall that earlier we looked at `tune.run` and we saw `search_alg.add_configurations([experiment])`. The implemention of `add_configurations` is a bit sophisticated but in our case (since we only pass one experiment but with `num_samples = N`) it amounts to setting `trials` above to be a list of length `num_samples` which each entry obtained from `create_trial_from_spec(experiment.spec)` (each entry gets a different "name"). This `create_trial_from_spec` is defined in [config_parser.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/config_parser.py) and returns an instance of the Trial class. So far this has **nothing to do with the class ImpalaTrainer**, the Trial just contains the words "IMPALA" and is wholly owned by the head node where the driver thread is running.

The class ImpalaTrainer becomes involved as [follows](https://github.com/ray-project/ray/blob/master/python/ray/tune/ray_trial_executor.py): in `ray_trial_executor.start_trial` listed above we find the line

```
trial.runner = self._setup_runner(trial,...)
```
and in turn `_setup_runner` either returns a cached runner or returns 
```
    def _setup_runner(self, trial, reuse_allowed):
        if (self._reuse_actors and reuse_allowed
                and self._cached_actor is not None):
            ...
        else:
            if self._cached_actor:
                ...
            existing_runner = None
            cls = ray.remote(
                num_cpus=trial.resources.cpu,
                num_gpus=trial.resources.gpu,
                resources=trial.resources.custom_resources)(
                    trial._get_trainable_cls())

        ...

        return cls.remote(config=trial.config, logger_creator=logger_creator)
```
So what does this mean? It means Ray will start an Actor with the specified resources and underlying class whatever the Trial is specifying (in our case "ImpalaTrainer" see [here](https://github.com/ray-project/ray/blob/master/python/ray/tune/trial.py)). Then the final line essentially runs `ImpalaTrainer()` i.e. the instance constructor, but does this with respect to the remote Actor, so that the returned instance is owned by whatever node in the cluster was determined by `ray.remote(...)`.

**SUMMARY:** instances of ImpalaTrainer are Ray actors, and they are attached to the trial objects as `trial.runner`.

Note that since `trial.resources.gpu` should be nonzero, when this Actor gets allocated to a node in the cluster, it should be allocated to the head machine (which is the only one we are putting GPUs) in. So **our trial.runner should be running on the head node** even though everything is executed through remote function calls.

## Impala actors as Ray actors of type PolicyEvaluator

In the [IMPALA implemention](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/impala/impala.py) we see (with some snips):

```
class ImpalaTrainer(Trainer):
    """IMPALA implementation using DeepMind's V-trace."""

    @override(Trainer)
    def _init(self, config, env_creator):
        self.local_evaluator = self.make_local_evaluator(self.env_creator, policy_cls)
        self.remote_evaluators = self.make_remote_evaluators(env_creator, policy_cls, config["num_workers"])
        self.optimizer = AsyncSamplesOptimizer(self.local_evaluator,self.remote_evaluators,...)
```
These functions `make_local_evaluator` and `make_remote_evaluators` are in [Trainer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py):

```
    def make_remote_evaluators(self, env_creator, policy_graph, count):
        """Convenience method to return a number of remote evaluators."""

        remote_args = {
            "num_cpus": self.config["num_cpus_per_worker"],
            "num_gpus": self.config["num_gpus_per_worker"],
            "resources": self.config["custom_resources_per_worker"],
        }

        cls = PolicyEvaluator.as_remote(**remote_args).remote

        return [
            self._make_evaluator(cls, env_creator, policy_graph, i + 1,
                                 self.config) for i in range(count)
        ]
        
    def make_local_evaluator(self,
                             env_creator,
                             policy_graph,
                             extra_config=None):
        """Convenience method to return configured local evaluator."""

        return self._make_evaluator(PolicyEvaluator,env_creator,policy_graph,...)
                
    def _make_evaluator(self, cls, env_creator, policy_graph, worker_index,config):
        def session_creator():
            return tf.Session(config=tf.ConfigProto(**config["tf_session_args"]))

        ...
        
        return cls(
            env_creator,
            policy_graph,
            tf_session_creator=(session_creator if config["tf_session_args"] else None),
            env_config=config["env_config"],
            model_config=config["model"])            
```
Note that the remote evaluators (IMPALA actors) are Ray actors, since their class has the `remote` at the end, whereas the local evaluator is an ordinary Python instance on the driver (head) node. The generation of rollouts using the remote evaluator, and the passing of sample data to the learner, is handled by the optimizer which is an instance of [AsyncSamplesOptimizer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/optimizers/async_samples_optimizer.py). But how do those remote IMPALA actors get the model weights when they need to be updated? Here we consult AsyncSamplesOptimizer:

```
class AsyncSamplesOptimizer(PolicyOptimizer):
    """Main event loop of the IMPALA architecture.
    This class coordinates the data transfers between the learner thread
    and remote evaluators (IMPALA actors).
    """

    def __init__(self,
                 local_evaluator,
                 remote_evaluators,
                 train_batch_size=500,
                 sample_batch_size=50,
                 ...
                 broadcast_interval=1,...):
        PolicyOptimizer.__init__(self, local_evaluator, remote_evaluators)

        self.learner = LearnerThread(self.local_evaluator,
                                         minibatch_buffer_size, num_sgd_iter,
                                         learner_queue_size)
        self.learner.start()

        self.aggregator = SimpleAggregator(
                self.local_evaluator,
                self.remote_evaluators,
                ...,
                train_batch_size=train_batch_size,
                sample_batch_size=sample_batch_size,
                broadcast_interval=broadcast_interval)
```
Note that this all happens on the same machine as the ImpalaTrainer actor, which is the head node (if that is the only one with GPUs). In particular the learner thread (which does gradient ops) is located there. This is the IMPALA learner, obviously. The IMPALA "actors" (that is, the experience gatherers) are in `self.remote_evaluators` and the communication between these gatherers and the learner is run through the [aggregator](https://github.com/ray-project/ray/blob/master/python/ray/rllib/optimizers/aso_aggregator.py), which reads

```
class SimpleAggregator(AggregationWorkerBase, Aggregator):
    """Simple single-threaded implementation of an Aggregator."""

    def __init__(self,
                 local_evaluator,
                 remote_evaluators,
                 max_sample_requests_in_flight_per_worker=2,
                 replay_proportion=0.0,
                 replay_buffer_num_slots=0,
                 train_batch_size=500,
                 sample_batch_size=50,
                 broadcast_interval=5):
        self.local_evaluator = local_evaluator
        self.broadcast_interval = broadcast_interval
        self.broadcast_new_weights()
        
        # Kick off async background sampling
        self.sample_tasks = TaskPool()
        for ev in self.remote_evaluators:
            ev.set_weights.remote(self.broadcasted_weights)
            for _ in range(max_sample_requests_in_flight_per_worker):
                self.sample_tasks.add(ev, ev.sample.remote())

        self.batch_buffer = []

    @override(Aggregator)
    def broadcast_new_weights(self):
        self.broadcasted_weights = ray.put(self.local_evaluator.get_weights())
        self.num_sent_since_broadcast = 0

    @override(Aggregator)
    def should_broadcast(self):
        return self.num_sent_since_broadcast >= self.broadcast_interval
```
There are several remote calls here: when the aggregator is initialised it puts `self.broadcasted_weights` onto the Ray communication system, sends it to all the remote evaluators (IMPALA actors) and uses `set_weights` on them with these broadcasted weights from the learner. The aggregator also makes a remote call of `sample` on each remote evaluator. Note that `self.broadcasted_weights` is an objectID generated by `broadcast_new_weights` also shown above, so the weights eventually come from the local evaluator (i.e. the learner).

**Question:** how does the sampling actually work? The call chain looks like this (where `{..}` denotes something wrapped in `try`):

```
Trainer.train() --> { Trainable.train() 
                        --> ImpalaTrainer._train() 
                        --> AsyncSamplesOptimizer.step() 
                        --> AsyncSamplesOptimizer._step()
                        --> AggregationWorkerBase.iter_train_batches()
                        --> AggregationWorkerBase._augment_with_replay()
                        --> sample_batch = ray_get_and_free(sample_batch) }
```
This final line is the thing that raises an exception if the remote evaluator is on a machine that has died, see the next section.

## Recovering from dead remote evaluators

From the above analysis we learn that while there are in principle many "remote" Ray actors involved in IMPALA/PBT training, in fact the ImpalaTrainer actors are all located on the head node. So the only Ray actors that are actually running on the remote machines are the experience gatherers (remote evaluators). This is why in the crash logs that we see from our experiments (in cases where a node goes offline) while the exception might caught at the level of `_process_result` it actually originates in a call that gets a sample batch from a remote evaluator (in our logs from Ray `0.6.5` this used to be `sample_batch = ray.get(sample_batch)` in `async_samples_optimizer.py` but now that functionality seems to be in `aso_aggregator.py` see above). 

The short of it is, that for IMPALA/PBT we expect that main exception we will see being thrown when a worker node leaves the cluster is from the `try` in `Trainer.train()` which looks like:

```
     for _ in range(1 + MAX_WORKER_FAILURE_RETRIES):
            try:
                result = Trainable.train(self)
            except RayError as e:
                if self.config["ignore_worker_failures"]:
                    logger.exception(
                        "Error in train call, attempting to recover")
                    self._try_recover()
                else:
                    logger.info(
                        "Worker crashed during call to train(). To attempt to "
                        "continue training without the failed worker, set "
                        "`'ignore_worker_failures': True`.")
                    raise e
            except Exception as e:
                time.sleep(0.5)  # allow logs messages to propagate
                raise e
            else:
                break
```
So we should turn on `ignore_worker_failures`. we also find in [Trainer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py)
```
def _try_recover(self):
        """Try to identify and blacklist any unhealthy workers.
        This method is called after an unexpected remote error is encountered
        from a worker. It issues check requests to all current workers and
        blacklists any that respond with error. If no healthy workers remain,
        an error is raised.
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
                ray_get_and_free(obj_id)
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

        self.optimizer.reset(healthy_evaluators)
```
which is inherited by ImpalaTrainer. This looks like what we want! Except that it doesn't seem to handle the case where the remote evaluator will not respond to remote function calls. Note that `optimizer.reset` gives us from [AsyncSamplesOptimizer](https://github.com/ray-project/ray/blob/master/python/ray/rllib/optimizers/async_samples_optimizer.py) has `async_samples_optimizer.reset(remote_evaluators)` 

```
    @override(PolicyOptimizer)
    def reset(self, remote_evaluators):
        self.remote_evaluators = remote_evaluators
        self.aggregator.reset(remote_evaluators)
```
**NOTE** As the above anlaysis indicates, turning on `ignore_worker_failures` seems to work, in the sense that it removes the failed workers from the `remote_evaluators` list of the AsyncSamplesOptimizer and the SimpleAggregator (although as far as we can tell, not from the ImpalaTrainer itself). However, ImpalaTrainer only seems to use `self.remote_evaluators` in the following places:
* during initialisation of ImpalaTrainer to pass this list to the optimizer
* a few places in `trainer.py`
So we want to edit `_try_recover` in Trainer to keep the `self.remote_evaluators` and `self.optimizer.remote_evaluators` in sync and also to fill back up the list of remote evaluators with new ones in the case where some have become unhealthy.

### Attempting reuse actors

When we set `reuse_actors = True` and hack the IMPALA implementation with

```
    def reset_config(self, new_config):
        self.config = new_config
        return True
```

# PBT

```
When the PBT scheduler is enabled, each trial variant is treated as a member of the population. Periodically,
top-performing trials are checkpointed (this requires your Trainable to support checkpointing). Low-performing
trials clone the checkpoints of top performers and perturb the configurations in the hope of discovering an
even better variation.```
```
The paper is Jaderberg et al "Population based training of neural networks". The key [code](https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pbt.py) is in `ray/tune/schedulers/pbt.py`. To understand this you have to first become familiar with the [Trial class](https://github.com/ray-project/ray/blob/master/python/ray/tune/trial.py) and for interactions with disk and memory the [Trial Executor class](https://github.com/ray-project/ray/blob/master/python/ray/tune/ray_trial_executor.py).

## Exploit and explore

From the class `PopulationBasedTraining` in `pbt.py` the main loop of PBT is contained in:

```
    def on_trial_result(self, trial_runner, trial, result):
        time = result[self._time_attr]
        state = self._trial_state[trial]

        if time - state.last_perturbation_time < self._perturbation_interval:
            return TrialScheduler.CONTINUE  # avoid checkpoint overhead

        score = result[self._reward_attr]
        state.last_score = score
        state.last_perturbation_time = time
        lower_quantile, upper_quantile = self._quantiles()

        if trial in upper_quantile:
            state.last_checkpoint = trial_runner.trial_executor.save(
                trial, Checkpoint.MEMORY)
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None  # not a top trial

        if trial in lower_quantile:
            trial_to_clone = random.choice(upper_quantile)
            assert trial is not trial_to_clone
            self._exploit(trial_runner.trial_executor, trial, trial_to_clone)

        for trial in trial_runner.get_trials():
            if trial.status in [Trial.PENDING, Trial.PAUSED]:
                return TrialScheduler.PAUSE  # yield time to other trials

        return TrialScheduler.CONTINUE
```

This is a callback that gets run every time a trial (that is, a member of the PBT population) completes an iteration. If the perturbation time has not elapsed then nothing happens. If it has elapsed, then there are three cases: depending on whether the trial is in the *top quantile* (by default this means 25%), *bottom quantile* or *other*. If the trial is in the top quantile then a **checkpoint is saved to memory**. If it is neither in the top nor bottom quantile nothing happens. If it is in the bottom quantile then a random member of the upper quantile is chosen to clone.

The `_exploit` function looks like:

```
    def _exploit(self, trial_executor, trial, trial_to_clone):
        """Transfers perturbed state from trial_to_clone -> trial."""

        trial_state = self._trial_state[trial]
        new_state = self._trial_state[trial_to_clone]
        if not new_state.last_checkpoint:
            logger.info("[pbt]: no checkpoint for trial."
                        " Skip exploit for Trial {}".format(trial))
            return
        new_config = explore(trial_to_clone.config, self._hyperparam_mutations,
                             self._resample_probability,
                             self._custom_explore_fn)
        logger.info("[exploit] transferring weights from trial "
                    "{} (score {}) -> {} (score {})".format(
                        trial_to_clone, new_state.last_score, trial,
                        trial_state.last_score))
        new_tag = make_experiment_tag(trial_state.orig_tag, new_config,
                                      self._hyperparam_mutations)
        reset_successful = trial_executor.reset_trial(trial, new_config,
                                                      new_tag)
        if reset_successful:
            trial_executor.restore(
                trial, Checkpoint.from_object(new_state.last_checkpoint))
        else:
            trial_executor.stop_trial(trial, stop_logger=False)
            trial.config = new_config
            trial.experiment_tag = new_tag
            trial_executor.start_trial(
                trial, Checkpoint.from_object(new_state.last_checkpoint))

        self._num_perturbations += 1
        # Transfer over the last perturbation time as well
        trial_state.last_perturbation_time = new_state.last_perturbation_time
```
Note that `new_config` is just hyperparameters, and is essentially the data we pass into `ray.tune`. This is printed out during an `[explore]` see below. So in `reset_trial` we are just copying hyperparameters, not mdoel weights: the model weights are copied by `restore` from the model checkpoint.

We have seen this `[pbt] no checkpoint for trial` quite frequently. There are a few ways this could happen. Note that a trial only saves a checkpoint if it is in the top quantile at the point where it reaches the `perturbation_interval` threshold in terms of the number of iterations. It's possible that a trial has just entered the top quantile, and not been there long enough to save a checkpoint. However, *as soon as it enters the top quantile* it is a valid target for being cloned. If there is a lot of churn in the population one would expect to see this happen (this is one reason why at least this particular implementation of PBT seems more rational in large populations). Note that if this happens, the trial that was scheduled to be executed and replaced by a perturbed clone of a top quantile trial, gets a reprieve and continues to live. So `[pbt] no checkpoint for trial` is nothing to worry about, unless it is happening for a significant proportion of all exploits.

**Remark.** Here is a potential bug: a trial enters the top quantile, stays there long enough to save a checkpoint, then drops out of the top quantile for 100 years, re-enters it for one iteration (long enough to show up as a potential target for other trials running `_exploit`) and then some other unsuspecting trial attempts to clone the 100 year old checkpoint. This either succeeds (in which case we clone a stale trial checkpoint into the current population) or fails (because for example the checkpoint data has passed out of the object store) and in either case is undesirable.

Note that when a trial is set to be replaced by a clone of a top quantile trial, it first tries `reset_trial`, then if that is successful `restore(checkpoint)` and otherwise it calls `stop_trial` and then `start_trial` (which contains within it `restore(checkpoint)`) where `checkpoint` is the hyperparameters and model weights of the trial to be cloned.

A working PBT exploit in our logs looks like (from `16-4-19-B`)

```
2019-04-16 10:58:34,637	INFO pbt.py:230 -- [exploit] transferring weights from trial
IMPALA_BoxWorld_3@perturbed[entropy_coeff=0.0080757,epsilon=0.1,lr=0.0027914] (score 5.733333333333333) ->
IMPALA_BoxWorld_0@perturbed[entropy_coeff=0.0084122,epsilon=0.1,lr=0.0029077] (score 5.11)
2019-04-16 10:59:05,515	WARNING util.py:62 -- The `reset_config` operation took 30.876898765563965 seconds to complete, which
may be a performance bottleneck.
2019-04-16 10:59:05,530	INFO ray_trial_executor.py:178 -- Destroying actor for trial
IMPALA_BoxWorld_0@perturbed[entropy_coeff=0.0064606,epsilon=1e-05,lr=0.0022331]. If your trainable is slow to initialize,
consider setting reuse_actors=True to reduce actor creation overheads.

```
Note that the `reset_config` warning comes from `ray_trial_executor.py` in the function `reset_trial` which is called as `reset_successful = trial_executor.reset_trial(trial, new_config,new_tag)` above. In this particular log, we can infer that `reset_successful` was false, as the code continued on to `stop_trial` (this is the only place we see a `Destroying actor for trial` error). So why should `reset_successful` be false? If we look in `ray_trial_executor.py` we see that it returns `reset_val = ray.get(trainable.reset_config.remote(new_config))` and so we have to understand `reset_config`.

That is a bit interesting, because the trainable we are passed here is `trial.runner` which is the IMPALA agent (which subclasses `Trainer` which subclasses `Trainable`). So we look in the IMPALA agent implementation [file](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/impala/impala.py) and find no `reset_config`, we look in the Trainer class [file](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py) and find no `reset_config` and we look in the Trainable class [file](https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable.py) and find that the stub implementation of `reset_config` returns FALSE. So in `_exploit` the boolean `reset_successful` is *always* false, at least for IMPALA (and as far as I can tell, none of the agents in the current version of RLlib implement `reset_config` so this is true for all of them). How that can take 30 seconds is a mystery to me, but I'm guessing that the `ray.get` and `remote(new_config)` wrappers actually propagate the configuration (hyperparameters and model weights, presumably) over the network, only to have this data then ignored by the stub implementation of `reset_config`, but this propagation still eats up time (30 seconds is still a very long time for that, though).

So in short, every time a trial goes through exploit, it will run `stop_trial` and `start_trial`. It seems straightforward to modify the IMPALA agent to support reset, see [this](https://github.com/ericl/ray/blob/1ce6f746bda802be03d54288ebaf7afa803cf0f2/python/ray/tune/tests/test_actor_reuse.py) and [this](https://github.com/ericl/ray/blob/1ce6f746bda802be03d54288ebaf7afa803cf0f2/python/ray/tune/examples/pbt_example.py). We should be a bit careful editing these files directly in case the pip3 version of ray changes, currently it is `0.6.5`.

And a successful explore looks like

```
2019-04-16 11:03:05,142	INFO pbt.py:81 -- [explore] perturbed config from {'env': <class '__main__.BoxWorld'>, 'model':
{'custom_model': 'my_model'}, 'monitor': False, 'min_iter_time_s': 30, 'sample_batch_size': 40, 'train_batch_size': 400,
'horizon': 30000.0, 'num_workers': 15, 'num_gpus': 0.5, 'log_level': 'WARN', 'opt_type': 'rmsprop', 'decay': 0.99, 
'momentum': 0.0, 'epsilon': 0.1, 'entropy_coeff': 0.008075742771959653, 'lr': 0.002791404243944746, 
'env_config': {'num_rows': 12, 'num_cols': 12, 'max_sol_length': 3, 'max_decoy_paths': 1, 'max_decoy_length': 1}} 
->
{'env': <class '__main__.BoxWorld'>, 'model': {'custom_model': 'my_model'}, 'monitor': False, 'min_iter_time_s': 30,
'sample_batch_size': 40, 'train_batch_size': 400, 'horizon': 30000.0, 'num_workers': 15, 'num_gpus': 0.5, 'log_level': 
'WARN', 'opt_type': 'rmsprop', 'decay': 0.99, 'momentum': 0.0, 'epsilon': 0.1, 'entropy_coeff': 0.006460594217567723, 
'lr': 0.003349685092733695, 'env_config': {'num_rows': 12, 'num_cols': 12, 'max_sol_length': 3, 'max_decoy_paths': 1,
'max_decoy_length': 1}}
```

This line `Destroying actor for trial` is worth understanding, as we see this every time a trial is replaced by exploit, as you can see above when we call `trial_executor.stop_trial`. This leads to the [code](https://github.com/ray-project/ray/blob/master/python/ray/tune/ray_trial_executor.py) `ray_trial_executor.py`
```
def _stop_trial(self, trial, error=False, error_msg=None,
                    stop_logger=True):
        """Stops this trial.
        Stops this trial, releasing all allocating resources. If stopping the
        trial fails, the run will be marked as terminated in error, but no
        exception will be thrown.
        Args:
            error (bool): Whether to mark this trial as terminated in error.
            error_msg (str): Optional error message.
            stop_logger (bool): Whether to shut down the trial logger.
        """

        if stop_logger:
            trial.close_logger()

        if error:
            self.set_status(trial, Trial.ERROR)
        else:
            self.set_status(trial, Trial.TERMINATED)

        try:
            trial.write_error_log(error_msg)
            if hasattr(trial, "runner") and trial.runner:
                if (not error and self._reuse_actors
                        and self._cached_actor is None):
                    logger.debug("Reusing actor for {}".format(trial.runner))
                    self._cached_actor = trial.runner
                else:
                    logger.info(
                        "Destroying actor for trial {}. If your trainable is "
                        "slow to initialize, consider setting "
                        "reuse_actors=True to reduce actor creation "
                        "overheads.".format(trial))
                    trial.runner.stop.remote()
                    trial.runner.__ray_terminate__.remote()
        except Exception:
            logger.exception("Error stopping runner for Trial %s", str(trial))
            self.set_status(trial, Trial.ERROR)
        finally:
            trial.runner = None
```

In `16-4-19-C` we encountered

```
2019-04-16 08:40:00,617	ERROR worker.py:1717 -- Possible unhandled error from worker: ray_ImpalaAgent:restore_from_object() (pid=23009, host=newton)
ray.exceptions.RayActorError: The actor died unexpectedly before finishing this task.

2019-04-16 08:40:22,982	ERROR worker.py:1780 -- The monitor failed with the following error:
Traceback (most recent call last):
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/monitor.py", line 383, in <module>
    monitor.run()
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/monitor.py", line 325, in run
    self.process_messages()
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/monitor.py", line 255, in process_messages
    message_handler(channel, data)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/monitor.py", line 220, in xray_driver_removed_handler
    self._xray_clean_up_entries_for_driver(driver_id)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/monitor.py", line 159, in _xray_clean_up_entries_for_driver
    task_table_object = task_info["TaskSpec"]
KeyError: 'TaskSpec'
```

```
2019-04-16 08:41:37,566	ERROR worker.py:1717 -- Possible unhandled error from worker: ray_ImpalaAgent:restore_from_object() (pid=22398, host=newton)
ray.exceptions.RayActorError: The actor died unexpectedly before finishing this task.
```
We probably understand this now. When a trial is set to be cloned during `exploit` it takes the checkpoint of the target, and tries `reset_trial`. In our cases, since we see the `Destroying actor for trial` message, that means we are in the logic branch where `start_trial` gets called. If you look in `ray_trial_executor.py` at `start_trial` it will call `restore` with the Checkpoint data, and then `trial.runner.restore_from_object.remote(value)`. If the checkpoint has passed out of the object store (I am not clear on whether this is on the head or the worker) then this fails with the exception that we have seen above. I.e. the `restore_from_object` error. Since increasing the size of the object store seems to fix this error, I think we can safely say we have understood it. One mitigation would be to check the age of the checkpoint and not attempt to clone from it if it is too old (let's say, `2 * perturbation_interval * min_iter_time_s`).

Also note that making the perturbation interval too long, or `min_iter_time_s` too long, will tend to exacerbate this problem because it means checkpoints are less frequent and thus more liable to fall out of the object store (which presumably contains other data, like the queue of samples for the learner).

# Keras, Tensorflow, etc

Keras is high-level library that serves as a common API to various backends, such as TensorFlow (TF) and Theano. We will use it exclusively with TF as a backend. Keras code can be mixed with TF code, with some care, see [this tutorial](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html). As the original author of Keras notes [here](https://github.com/keras-team/keras/issues/3223)

> When you are using the TensorFlow backend, your Keras code is actually building a TF graph. You can just grab this graph. Keras only uses one graph and one session. You can access the session via: `K.get_session()`. The graph associated with it would then be: `K.get_session().graph`.

Here `K` refers to what you get from running `from keras import backend as K`. You can just read the Keras source to see how this works, for example `keras/backend/tensorflow_backend.py` [here](https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py).

This is good news, because RLlib requires agents to be written as TensorFlow models, so if we continue to use Keras to generate the agent model (CNN + Transformer) then we will need to pass the resulting TensorFlow graph to RLlib, as it doesn't directly understand Keras classes.

NOTE: Keras-transformer requires Python 3.

If you're confused about shapes see [this](http://www.heyuhang.com/blog/2018/07/14/tensorflow-get-shape-vs-tf-dot-shape/).

# Understanding RLlib

To distribute the Agent policy and Environment to the workers, Python [pickle](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled) is used, and this has some limitations you might want to be aware of (most importantly, all the modules used to construct the agent and environment need also to be importable in the python3 on the worker machine).

The [Tune user guide](https://ray.readthedocs.io/en/latest/tune-usage.html) is a useful place to start, in understanding the logging output you will see during training (see also how to use Tensorboard). For information on curriculum learning and using OpenAI Gym Monitors (videos) see [this](https://ray.readthedocs.io/en/latest/rllib-training.html).

Some statistics from one of our early runs, with the IMPALA defaults, from the RLlib logs:

```
episode_len_mean: 566.1239669421487
episode_reward_max: -48.0
episode_reward_mean: -536.1239669421487
episode_reward_min: -2066.0
episodes_this_iter: 121
train_throughput: 6873.344
time_this_iter_s: 10.387565851211548
timesteps_this_iter: 70500
```

So some terms that need explaining are *iteration*, *episode*, *timesteps*. A timestep means an environment timestep in the usual sense (i.e. one agent action and observation). For the definition of an iteration see below. An *episode* is the interval between `done` being False and True (this flag is controlled in our Environment's step function), i.e. an episode of the game in the usual sense. You can see above that there were `121` episodes in the iteration.

Here are some of the [agent defaults](https://ray.readthedocs.io/en/latest/rllib-training.html#specifying-resources)

```
# Default sample batch size (unroll length). Batches of this size are
# collected from workers until train_batch_size is met. When using
# multiple envs per worker, this is multiplied by num_envs_per_worker.

"sample_batch_size": 200,

# Training batch size, if applicable. Should be >= sample_batch_size.
# Samples batches will be concatenated together to this size for training.

"train_batch_size": 200,

# Whether to rollout "complete_episodes" or "truncate_episodes"
"batch_mode": "truncate_episodes"
```

The batch_mode is documented more fully in the implementation of the [policy evaluator](https://github.com/ray-project/ray/blob/master/python/ray/rllib/evaluation/policy_evaluator.py) along with the important flag `episode_horizon`:

```
# batch_steps (int): The target number of env transitions to include
#                in each sample batch returned from this evaluator.
# batch_mode (str): One of the following batch modes:
#                "truncate_episodes": Each call to sample() will return a batch
#                    of at most `batch_steps * num_envs` in size. The batch will
#                    be exactly `batch_steps * num_envs` in size if
#                    postprocessing does not change batch sizes. Episodes may be
#                    truncated in order to meet this size requirement.
#                "complete_episodes": Each call to sample() will return a batch
#                    of at least `batch_steps * num_envs` in size. Episodes will
#                    not be truncated, but multiple episodes may be packed
#                    within one batch to meet the batch size. Note that when
#                    `num_envs > 1`, episode steps will be buffered until the
#                    episode completes, and hence batches may contain
#                    significant amounts of off-policy data.
#
# episode_horizon (int): Whether to stop episodes at this horizon.
```

The IMPALA defaults are a little different than the generic Agent defaults above:

```
"sample_batch_size": 50,
"train_batch_size": 500
```

NOTE that IMPALA *requires* `"batch_mode": "truncate_episodes"` (see vtrace_policy_graph.py). So my understanding is that, if an episode takes 500 timesteps (not unusual) then its transitions will be spread over 10 sample batches for processing. Note the episode is not *terminated at 50 timesteps*. That seems to be what `episode_horizon` is for.

NOTE: `sample_batch_size` is a synonym for *unroll length* (In the RLlib docs they say this) which is a synonym for the integer n in n-step V-trace as defined in Section 4.1 of the IMPALA paper. This we can tell because they write Unroll length (n) in their tables, e.g. Table D.3. FOr example “unroll length (n) = 20” for the Atari environments. In the Zambaldi paper they say “unroll length = 40” for BoxWorld. , and the defaults are consistent with these kinds of numbers.

## What is an iteration?

By definition an iteration is one step of Tune running train() on a Trainable, which in our case is an IMPALA agent. This boils down to the following [code](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/impala/impala.py):

```
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
```
Earlier `min_iter_time_s` is defined by default to be 10 seconds. So in practice, an iteration is just 10 seconds, and we can see that in our logs that is correct, for example `time_this_iter_s: 10.790229320526123`. I think that the distribution of weights from the learner to the workers happens once per iteration.

# IMPALA

Most of what is happening in an RLlib agent, say IMPALA, is happening in the base Agent class here:
https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/agent.py and what is interesting there is the function compute_action, which includes

```
if state is None:
    state = []
    preprocessed = self.local_evaluator.preprocessors[policy_id].transform(observation)
    filtered_obs = self.local_evaluator.filters[policy_id](preprocessed, update=False)
if state:
    return self.get_policy(policy_id).compute_single_action(
        filtered_obs,
        state,
        prev_action,
        prev_reward,
        info,
        clip_actions=self.config["clip_actions"])
 return self.get_policy(policy_id).compute_single_action(
    filtered_obs,
    state,
    prev_action,
    prev_reward,
    info,
    clip_actions=self.config["clip_actions"])[0]
```

note that `get_policy(policy_id)` amounts to `self.local_evaluator.get_policy(policy_id)` and local_evaluator comes from `_init` where we see `compute_single_action` is in `ray/rllib/evaluation/policy_graph.py`.

# Understanding scaling

Note that the CPUs a trial is using may be spread over multiple machines in the cluster. Hence if one node drops out, many trials may be affected. We have done some brief experiments that indicate that Ray can adapt smartly to this situation. It attempts to recover each trial from its latest checkpoint, and then uses the remaining CPUs to continue running. If the remaining CPUs are no longer enough for each trial to run simultaneously, they will begin multi-plexing (that is, some of them will be in a PENDING state while others are RUNNING). If the node then rejoins the cluster, the extra CPUs will be used again restoring the original training situation. This is very nice software!

# Statistics

The key statistics are formed in the [code](https://github.com/ray-project/ray/blob/master/python/ray/rllib/optimizers/policy_optimizer.py) from `ray/rllib/optimizers/policy_optimizer.py`:

```
def collect_metrics(self,
                        timeout_seconds,
                        min_history=100,
                        selected_evaluators=None):
        """Returns evaluator and optimizer stats.

        Arguments:
            timeout_seconds (int): Max wait time for a evaluator before
                dropping its results. This usually indicates a hung evaluator.
            min_history (int): Min history length to smooth results over.
            selected_evaluators (list): Override the list of remote evaluators
                to collect metrics from.

        Returns:
            res (dict): A training result dict from evaluator metrics with
                `info` replaced with stats from self.
        """
        episodes, num_dropped = collect_episodes(
            self.local_evaluator,
            selected_evaluators or self.remote_evaluators,
            timeout_seconds=timeout_seconds)
        orig_episodes = list(episodes)
        missing = min_history - len(episodes)
        if missing > 0:
            episodes.extend(self.episode_history[-missing:])
            assert len(episodes) <= min_history
        self.episode_history.extend(orig_episodes)
        self.episode_history = self.episode_history[-min_history:]
        res = summarize_episodes(episodes, orig_episodes, num_dropped)
        res.update(info=self.stats())
        return res
```
And `summarize_episodes` is defined in the [code](https://github.com/ray-project/ray/blob/master/python/ray/rllib/evaluation/metrics.py) `ray/rllib/evaluation/metrics.py`:

```
def summarize_episodes(episodes, new_episodes, num_dropped):
    """Summarizes a set of episode metrics tuples.

    Arguments:
        episodes: smoothed set of episodes including historical ones
        new_episodes: just the new episodes in this iteration
        num_dropped: number of workers haven't returned their metrics
    """

    if num_dropped > 0:
        logger.warning("WARNING: {} workers have NOT returned metrics".format(
            num_dropped))

    episodes, estimates = _partition(episodes)
    new_episodes, _ = _partition(new_episodes)

    episode_rewards = []
    episode_lengths = []
    policy_rewards = collections.defaultdict(list)
    custom_metrics = collections.defaultdict(list)
    perf_stats = collections.defaultdict(list)
    for episode in episodes:
        episode_lengths.append(episode.episode_length)
        episode_rewards.append(episode.episode_reward)
        for k, v in episode.custom_metrics.items():
            custom_metrics[k].append(v)
        for k, v in episode.perf_stats.items():
            perf_stats[k].append(v)
        for (_, policy_id), reward in episode.agent_rewards.items():
            if policy_id != DEFAULT_POLICY_ID:
                policy_rewards[policy_id].append(reward)
    if episode_rewards:
       min_reward = min(episode_rewards)
        max_reward = max(episode_rewards)
    else:
        min_reward = float('nan')
        max_reward = float('nan')
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    for policy_id, rewards in policy_rewards.copy().items():
        policy_rewards[policy_id] = np.mean(rewards)

    for k, v_list in custom_metrics.copy().items():
        custom_metrics[k + "_mean"] = np.mean(v_list)
        filt = [v for v in v_list if not np.isnan(v)]
        if filt:
            custom_metrics[k + "_min"] = np.min(filt)
            custom_metrics[k + "_max"] = np.max(filt)
        else:
            custom_metrics[k + "_min"] = float("nan")
            custom_metrics[k + "_max"] = float("nan")
        del custom_metrics[k]

    for k, v_list in perf_stats.copy().items():
        perf_stats[k] = np.mean(v_list)

    estimators = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in estimates:
        acc = estimators[e.estimator_name]
        for k, v in e.metrics.items():
            acc[k].append(v)
    for name, metrics in estimators.items():
        for k, v_list in metrics.items():
            metrics[k] = np.mean(v_list)
        estimators[name] = dict(metrics)

    return dict(
        episode_reward_max=max_reward,
        episode_reward_min=min_reward,
        episode_reward_mean=avg_reward,
        episode_len_mean=avg_length,
        episodes_this_iter=len(new_episodes),
        policy_reward_mean=dict(policy_rewards),
        custom_metrics=dict(custom_metrics),
        sampler_perf=dict(perf_stats),
        off_policy_estimator=dict(estimators),
        num_metric_batches_dropped=num_dropped)
```

## Custom metric

In our original experiments with PBT we wanted the comparison of trials to be done on *negative* `episode_len_mean` for which we need a custom metric (**note**: this is no longer used, but kept here for reference). These are not really supported currently, so we needed to hack our own copy of `~/.local/lib/python3.6/site-packages/ray/tune/schedulers/pbt.py` to have

```
    def on_trial_result
        ...
        score = result["custom_metrics"]["pbt_metric_mean"]
        # score = result[self._reward_attr]
```

here `self._reward_attr` is whatever string we set above. So it is reading into the result dictionary which is in turn provided in `_process_trial` and thus in turn by the `trial_executor.fetch_result`. This modified `pbt.py` is in the GitHub repo and among the install instructions there are instructions for copying this to the relevant directory on all machines. See `cp ~/simplicialtransformer/python/pbt.py ~/.local/lib/python3.6/site-packages/ray/tune/schedulers/`.

# Episode horizon

The "horizon" flag that may be passed to an agent ends up in the [code](https://github.com/ray-project/ray/blob/master/python/ray/rllib/evaluation/sampler.py) `~/.local/lib/python3.6/site-packages/ray/rllib/evaluation/sampler.py` in the function `_env_runner`:

```
        # Check episode termination conditions
        if dones[env_id]["__all__"] or episode.length >= horizon:
            all_done = True
            atari_metrics = _fetch_atari_metrics(base_env)
            if atari_metrics is not None:
                for m in atari_metrics:
                    outputs.append(
                        m._replace(custom_metrics=episode.custom_metrics))
            else:
                outputs.append(
                    RolloutMetrics(episode.length, episode.total_reward,
                                   dict(episode.agent_rewards),
                                   episode.custom_metrics, {}))
        else:
            all_done = False
            active_envs.add(env_id)
```
We have verified that with IMPALA this actually works and imposes the horizon.

# Memory errors

```
2019-04-14 08:45:43,989	ERROR trial_runner.py:460 -- Error processing event.
Traceback (most recent call last):
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/trial_runner.py", line 409, in _process_trial
    result = self.trial_executor.fetch_result(trial)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/ray_trial_executor.py", line 314, in fetch_result
    result = ray.get(trial_future[0])
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/worker.py", line 2316, in get
    raise value
ray.exceptions.RayTaskError: ray_ImpalaAgent:train() (pid=19329, host=newton)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/memory_monitor.py", line 77, in raise_if_low_memory
    self.error_threshold))
ray.memory_monitor.RayOutOfMemoryError: More than 95% of the memory on node newton is used (60.21 / 63.32 GB). The top 5 memory consumers are:

PID	MEM	COMMAND
19329	20.56GB	ray_ImpalaAgent:train()
19327	20.55GB	ray_ImpalaAgent:train()
19326	20.52GB	ray_ImpalaAgent:train()
19311	19.1GB	/home/murfetd/.local/lib/python3.6/site-packages/ray/core/src/ray/raylet/raylet /tmp/ray/session_201
2441	13.75GB	/usr/bin/python3 /home/murfetd/.local/bin/tensorboard --logdir=~/ray_results

In addition, ~19.13 GB of shared memory is currently being used by the Ray object store. You can set the object store size with the `object_store_memory` parameter when starting Ray, and the max Redis size with `redis_max_memory`.
```

For some time we were getting `restore_from_object` errors:

```
2019-04-15 03:43:06,950	ERROR worker.py:1717 -- Possible unhandled error from worker: ray_ImpalaAgent:restore_from_object() (pid=12427, host=newton)
ray.exceptions.UnreconstructableError: Object 010000002e9ebaf154673216295811ebbe63f7ce is lost (either evicted or explicitly deleted) and cannot be reconstructed.
```

This happens while trying to transfer weights during PBT, so apparently PBT is storing its weights for checkpoints in memory (I think I recall seeing that in the pbt code, as well) and by capping the object store we are prematurely nuking those checkpoints. Thisis was fixed by manually setting the object store memory to be at least `30Gb` by e.g. `ray start --head --redis-port=6379 --num-cpus=12 --num-gpus=2 --object-store-memory 30000000000`.

# Transformer

Some notes on [this talk](https://www.youtube.com/watch?v=5vcj8kSwBCY&feature=youtu.be) by Vaswani.

* The residual connections are transmitting the positional information through the network, and are necessary for the attention mechanism to learn the diagonal.

# Videos

The video code uses OpenAI Gym wrappers, see

https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py 
https://github.com/ray-project/ray/blob/master/python/ray/rllib/evaluation/policy_evaluator.py
https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/trainer.py

The trainer.py sets the monitor_path=self.logdir if config["monitor"] else None, then this is read by policy_evaluator.py which replaces the environment by env = _monitor(env, monitor_path) where

```
def _monitor(env, path):
  return gym.wrappers.Monitor(env, path, resume=True)
```

see also https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py. We have edited `~/.local/lib/python3.6/site-packages/ray/rllib/evaluation/policy_evaluator.py` on both Dirac and Clifford to replace `_monitor` by

```
def _monitor(env, path):
  from gym.wrappers import Monitor
  return Monitor(env, path, video_callable=lambda episode_id: episode_id%200 == 0, resume=True)
```

NOTE: filenames are the output of

```
os.path.join(self.directory, '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id))
```

NOTE: each of the 64 workers maintains its own episode_id count, so when video_callable evaluates to true we get many workers writing videos. If we take episode_id%200 then we get videos after a total of roughly 22,000 episodes, which amounts to ~25M timesteps (at 2000 timesteps per episode)

# Stability of Ray with pre-emptible instances

Currently we train with a large cluster of pre-emptible GCP instances, which can drop out of the Ray network at any moment without notice. This leads to various errors, which hopefully are caught as exceptions and handled by Ray. Depending on what operation Ray is doing when the node becomes inaccessible these errors vary, for instance:

## Error type 1
```
2019-04-18 01:39:23,578	ERROR trial_runner.py:460 -- Error processing event.
Traceback (most recent call last):
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/trial_runner.py", line 409, in _process_trial
    result = self.trial_executor.fetch_result(trial)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/ray_trial_executor.py", line 314, in fetch_result
    result = ray.get(trial_future[0])
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/worker.py", line 2316, in get
    raise value
ray.exceptions.RayTaskError: ray_ImpalaAgent:train() (pid=7149, host=newton)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/agents/agent.py", line 316, in train
    raise e
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/agents/agent.py", line 305, in train
    result = Trainable.train(self)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/trainable.py", line 151, in train
    result = self._train()
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/agents/impala/impala.py", line 124, in _train
    self.optimizer.step()
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/optimizers/async_samples_optimizer.py", line 141, in step
    sample_timesteps, train_timesteps = self._step()
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/optimizers/async_samples_optimizer.py", line 190, in _step
    self.sample_tasks.completed_prefetch()):
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/optimizers/async_samples_optimizer.py", line 238, in _augment_with_replay
    sample_batch = ray.get(sample_batch)
ray.exceptions.RayActorError: The actor died unexpectedly before finishing this task.
2019-04-18 01:39:24,498	INFO ray_trial_executor.py:178 -- Destroying actor for trial IMPALA_BoxWorld_5. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-18 01:39:24,501	INFO trial_runner.py:497 -- Attempting to recover trial state from last checkpoint.
```
In this case in the [code](https://github.com/ray-project/ray/blob/master/python/ray/tune/trial_runner.py) `trial_runner.py` we are in `_process_trial` and an exception is caught while running `fetch_result` which leads to the handling code:

```
logger.exception("Error processing event.")
error_msg = traceback.format_exc()
if trial.status == Trial.RUNNING:
    if trial.should_recover():
        self._try_recover(trial, error_msg)
    else:
        self._scheduler_alg.on_trial_error(self, trial)
        self._search_alg.on_trial_complete(
              trial.trial_id, error=True)
              self.trial_executor.stop_trial(
              trial, error=True, error_msg=error_msg)
```
and `_try_recover` looks like
```
    def _try_recover(self, trial, error_msg):
        """Tries to recover trial.
        Notifies SearchAlgorithm and Scheduler if failure to recover.
        Args:
            trial (Trial): Trial to recover.
            error_msg (str): Error message from prior to invoking this method.
        """
        try:
            self.trial_executor.stop_trial(
                trial,
                error=error_msg is not None,
                error_msg=error_msg,
                stop_logger=False)
            trial.result_logger.flush()
            if self.trial_executor.has_resources(trial.resources):
                logger.info("Attempting to recover"
                            " trial state from last checkpoint.")
                self.trial_executor.start_trial(trial)
                if trial.status == Trial.ERROR:
                    raise RuntimeError("Trial did not start correctly.")
            else:
                logger.debug("Notifying Scheduler and requeueing trial.")
                self._requeue_trial(trial)
        except Exception:
            logger.exception("Error recovering trial from checkpoint, abort.")
            self._scheduler_alg.on_trial_error(self, trial)
            self._search_alg.on_trial_complete(trial.trial_id, error=True)
```
The call to `trial_executor.stop_trial` is what we see in the log, and then we see the notice about attempting to recover the trial from a checkpoint. What happens then is a little non-obvious. We call `trial_executor.start_trial(trial)` with no checkpoint, which results in trying `self._start_trial(trial, None)` which results in `self.restore(trial, None)` which results in
```
    def restore(self, trial, checkpoint=None):
        """Restores training state from a given model checkpoint.
        This will also sync the trial results to a new location
        if restoring on a different node.
        """
        if checkpoint is None or checkpoint.value is None:
            checkpoint = trial._checkpoint
        if checkpoint is None or checkpoint.value is None:
            return True
        if trial.runner is None:
            logger.error("Unable to restore - no runner.")
            self.set_status(trial, Trial.ERROR)
            return False
        try:
            value = checkpoint.value
            if checkpoint.storage == Checkpoint.MEMORY:
                assert type(value) != Checkpoint, type(value)
                trial.runner.restore_from_object.remote(value)
            else:
                worker_ip = ray.get(trial.runner.current_ip.remote())
                trial.sync_logger_to_new_location(worker_ip)
                with warn_if_slow("restore_from_disk"):
                    ray.get(trial.runner.restore.remote(value))
            trial.last_result = checkpoint.last_result
            return True
        except Exception:
            logger.exception("Error restoring runner for Trial %s.", trial)
            self.set_status(trial, Trial.ERROR)
            return False
```
since we passed `checkpoint = None` checkpoint is immediately replaced by `trial._checkpoint`. Note that this is what is written to when a checkpoint is created (these checkpoints are separate from PBT checkpoints, so these are the ones whose frequency is controlled by `checkpoint_freq` (currently every 50 iterations), as we see in the following snippet of `save` from `ray_trial_executor.py`:

```
def save(self, trial, storage=Checkpoint.DISK):
        """Saves the trial's state to a checkpoint."""
        trial._checkpoint.storage = storage
        trial._checkpoint.last_result = trial.last_result
        trial._checkpoint.value = ray.get(trial.runner.save.remote())
```
So this is as expected: if a node leaves the Ray cluster then Ray will attempt to restore any affected trials from the most recent checkpoint on disk.

## Error type 2

```
2019-04-18 02:11:56,294	ERROR worker.py:1780 -- The node with client ID 37669ef11ddbd4e5fb6ae9e089dedc0475c3f89d has been marked dead because the monitor has missed too many heartbeats from it.
2019-04-18 02:11:57,317	ERROR trial_runner.py:460 -- Error processing event.
Traceback (most recent call last):
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/trial_runner.py", line 409, in _process_trial
    result = self.trial_executor.fetch_result(trial)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/ray_trial_executor.py", line 314, in fetch_result
    result = ray.get(trial_future[0])
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/worker.py", line 2316, in get
    raise value
ray.exceptions.RayTaskError: ray_ImpalaAgent:train() (pid=11515, host=newton)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/agents/agent.py", line 316, in train
    raise e
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/agents/agent.py", line 305, in train
    result = Trainable.train(self)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/tune/trainable.py", line 151, in train
    result = self._train()
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/agents/impala/impala.py", line 125, in _train
    result = self.collect_metrics()
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/agents/agent.py", line 571, in collect_metrics
    selected_evaluators=selected_evaluators)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/optimizers/policy_optimizer.py", line 133, in collect_metrics
    timeout_seconds=timeout_seconds)
  File "/home/murfetd/.local/lib/python3.6/site-packages/ray/rllib/evaluation/metrics.py", line 42, in collect_episodes
    metric_lists = ray.get(collected)
ray.exceptions.RayActorError: The actor died unexpectedly before finishing this task.

2019-04-18 02:11:57,337	INFO ray_trial_executor.py:178 -- Destroying actor for trial IMPALA_BoxWorld_6@perturbed[entropy_coeff=0.006,epsilon=0.001,lr=0.0032309]. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
```
We see this repeated for several other trials.

# How checkpoints work

We may not be correctly implementing `_restore` and `_save` for checkpoints, as described [here](https://ray.readthedocs.io/en/latest/tune-usage.html) under "Trial checkpointing". In particular we do not attempt to save TF sessions as in [this example](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tune_mnist_ray_hyperband.py) and we're not sure if that is done anywhere in the code we're currently using. This is one potential explanation for our problems with checkpointing on pre-emptible instances.

# Layer normalisation

Here we examine how layer normalisation is done in Keras Transformer (and our code) vs how it is done in recent iterations of the Transformer in the literature and in the library [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). Currently we are using Keras Transformer's LayerNormalization which is the Keras layer

```
class LayerNormalization(Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    "Unlike batch normalization, layer normalization performs exactly
    the same computation at training and test times."
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result
```
Note all biases are initialised to zero and all gains to 1 (good?). So this makes sense. Now, both the original Transformer paper ("Attention is all you need") and the Universal Transformer operate the transformer block in the following way, starting at the very first layer with input embedding and position encoding, then repetitions of:
```
H' = norm[norm(H + a(H)) + b(H)]
a(H) = dropout(attention(H))
b(H) = dropout(ff(norm(H + a(H))))
```
The OpenAI [Sparse Transformer](https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py#L13) does the following (see eqs.9-14) beginning with embedded inputs
```
H' = H + a(H) + b(H)
a(H) = dropout(attention(norm(H)))
b(H) = dropout(ff(norm(H + a(H))))
```
Note the differences in the organisation of the normalisations, and also the skip connections. 

We can see the actual code for the implementation of the "standard" Transformer in Keras Transformer's [TransformerBlock](https://github.com/kpot/keras-transformer/blob/master/keras_transformer/transformer.py) layer:

```
    def __call__(self, _input):
        output = self.attention_layer(_input)
        post_residual1 = (
            self.addition_layer([_input, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(self.addition_layer([_input, output])))
        norm1_output = self.norm1_layer(post_residual1)
        output = self.transition_layer(norm1_output)
        post_residual2 = (
            self.addition_layer([norm1_output, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(
                self.addition_layer([norm1_output, output])))
        output = self.norm2_layer(post_residual2)
        return output
```
and the actual code for the sparse Transformer at the above link (irrelevant bits snipped), although this seems to be missing dropout (?):
```
def transformer_block(x, scope, train=False):
    """
    core component of transformer
    performs attention + residual mlp + layer normalization
    """
    n_state = x.shape[-1].value

    with tf.variable_scope(scope):

        h = layernorm(x, "norm_a")

        q = conv1d(h, 'proj_q', n_state)
        k = conv1d(h, 'proj_k', n_state)
        v = conv1d(h, 'proj_v', n_state)

        # only need to create one bst per config
        # we could pass this in as an external param but I like to keep the code more local
        bst_params = (hps.n_timesteps, hps.n_head)
        bst = bst_cache.get(bst_params)
        if bst is None:
            bst = bst_cache[bst_params] = get_blocksparse_transformer(*bst_params)

        # run the core bst ops, transposes for dealing with heads are fused in here.
        w = bst.query_key_op(q, k)
        w = bst.masked_softmax(w, scale=1.0/np.sqrt(n_state / hps.n_head))
        a = bst.weight_value_op(w, v)

        a = conv1d(a, 'proj_a', n_state, std=0.02/hps.n_layer)

        # many basic tf ops are about half as fast as they should be in fp16
        x = bs.add(x, a)

        m = layernorm(x, "norm_m")

        # fast_gelu: x * sigmoid(1.702 * x)
        m = conv1d(m, 'proj_m1', n_state * hps.mlp_ratio, fast_gelu=True)
        m = conv1d(m, 'proj_m2', n_state)

        return bs.add(x, m)
```
**Question:** Is there a difference between Conv1D and a dense layer operating on the final dimension of the tensor? In any case, we clearly see in the Sparse Transformer code that they take the layer norm and then project to get query, key and value vectors.

In v6 of the notebook we are doing something else again, namely
```
H' = norm[H + ff(attention(H))]
```
Note both the Keras Transformer and Sparse Transformer use `gelu` for their transition feedforward networks.

# Ray debugging

See the [troubleshooting guide](https://ray.readthedocs.io/en/latest/troubleshooting.html) and [development tips](https://ray.readthedocs.io/en/latest/development.html). The errors we have seen with `restore_from_object` which we fixed by increasing the size of the object store are described under "Lost objects" [here](https://ray.readthedocs.io/en/latest/fault-tolerance.html).

> When an actor dies (either because the actor process crashed or because the node that the actor was on died), by default any attempt to get an object from that actor that cannot be created will raise an exception. Subsequent releases will include an option for automatically restarting actors.

We were originally also getting errors like the following:
```
TuneError: There are paused trials, but no more pending trials with sufficient resources.
```
but this is because you need to assign workers and GPUs *per trial*, so if you have 100 CPUs and population size 2, you would allocate 50 CPUs to each trial, so `num_workers=50`. 

**NOTE**: By default we are probably pickling TF graphs to pass them around with Ray, this might not be smart, see [this](https://ray.readthedocs.io/en/latest/using-ray-with-tensorflow.html) and [this](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tune_mnist_ray_hyperband.py).

NOTE: the Ray autoscaler files all run `ulimit -n 65536` before starting Ray see [this](https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/aws/example-full.yaml). You can see why [here](https://ray.readthedocs.io/en/latest/troubleshooting.html).

# Learning algorithms

See [this](https://gist.github.com/dmurfet/d3f029f5c972725b1117691564d920d0).

RMSProp vs Adam: DeepMind uses RMSProp in most of their RL work (e.g. the Mnih atari paper the IMPALA paper, and Zambaldi et al). There is some [speculation online](https://www.quora.com/Why-does-Google-DeepMind-use-RMSProp-over-Adam-for-RL) that this is because RMSProp is better at non-stationary data, but who knows.

# TensorFlow sessions

Our models are loaded inside policy evaluators, and inside the `__init__` method for the class [PolicyEvaluator](https://github.com/ray-project/ray/blob/master/python/ray/rllib/evaluation/policy_evaluator.py) we find

```
                    self.tf_sess = tf.Session(
                        config=tf.ConfigProto(
                            gpu_options=tf.GPUOptions(allow_growth=True)))
                with self.tf_sess.as_default():
                    self.policy_map, self.preprocessors = \
                        self._build_policy_map(policy_dict, policy_config)
```
This means that anywhere in our code that is touching TensorFlow, we should be inside that `with self.tf_sess.as_default()` and we can call `tf.get_default_graph()` to recover the global TF session. For example in [VTracePolicyGraph](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/impala/vtrace_policy_graph.py) we see `self.sess = tf.get_default_session()`. 

# Re-use actors

To support `reuse_actors=True` we have edited `~/.local/lib/python3.6/site-packages/ray/rllib/agents/impala/impala.py` to include

```
import copy

def reset_config(self, new_config):
   config = copy.deepcopy(DEFAULT_CONFIG)
   config.update(new_config)
```

# Tensorflow

Use `tf.einsum`! See [this blog post](https://rockt.github.io/2018/04/30/einsum).

# Resuming experiments

In `ray/tune/trial_runner.py` you will find that when resuming Ray will look in `ray_results/IMPALA` for the latest checkpoint file (meaning files of the form `experiment_state-2019-05-25_12-57-24.json`) and then load from it. For example the experiment `30-7-19-C` corresponds to `experiment_state-2019-07-29_22-01-38.json` inside which you can find binary data and the specifications of the experiment

```
"logdir": "/home/ubuntu/ray_results/IMPALA/IMPALA_BoxWorld_1_2019-07-29_22-01-399acvl93u",
"env_config": {
            "num_rows": 7,
            "num_cols": 9,
            "min_sol_length": 1,
            "max_sol_length": 3,
            "max_decoy_paths": 0,
            "max_decoy_length": 1,
            "multi_lock": true,
            "has_bridge": true,
            "episode_horizon": -1,
            "monitor_interval": 800
          },
```
Make sure you verify that this matches the experiment you want to resume. Then touch that experiment_state file to make it most recent timestamp, and rename (if necessary) the checkpoint folder to the logdir specified in experiment_state. Then just resume the experiment as normal. Be careful that the experiment state file may contain multiple trials, in which case you need to prepare the checkpoint folder for both of them. It seems likely the checkpoint it will try to restore is indicated by the lines

```
"training_iteration": 31650,
```

so make sure those checkpoints are available.
