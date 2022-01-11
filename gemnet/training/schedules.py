import torch
from torch.optim.lr_scheduler import LambdaLR
import logging
import numpy as np

class LinearWarmupExponentialDecay(LambdaLR):
    """This schedule combines a linear warmup with an exponential decay.

    Parameters
    ----------
        optimizer: Optimizer
            Optimizer instance.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay.
        last_step: int
            Only needed when resuming training to resume learning rate schedule at this step.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase=False,
        last_step=-1,
        verbose=False,
    ):
        assert decay_rate <= 1

        if warmup_steps == 0:
            warmup_steps = 1

        def lr_lambda(step):
            # step starts at 0
            warmup = min(1 / warmup_steps + 1 / warmup_steps * step, 1)
            exponent = step / decay_steps
            if staircase:
                exponent = int(exponent)
            decay = decay_rate ** exponent
            return warmup * decay

        super().__init__(optimizer, lr_lambda, last_epoch=last_step, verbose=verbose)

class MultiWrapper:
    def __init__(self, *ops):
        self.wrapped = ops

    def __getitem__(self, idx):
        return self.wrapped[idx]

    def zero_grad(self):
        for op in self.wrapped:
            op.zero_grad()

    def step(self):
        for op in self.wrapped:
            op.step()

    def state_dict(self):
        """Returns the overall state dict of the wrapped instances."""
        return {i: opt.state_dict() for i, opt in enumerate(self.wrapped)}

    def load_state_dict(self, state_dict):
        """Load the state_dict for each wrapped instance.
        Assumes the order is the same as when the state_dict was loaded
        """
        for i, opt in enumerate(self.wrapped):
            opt.load_state_dict(state_dict[i])


class ReduceLROnPlateau:
    """Reduce learning rate (and weight decay) when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of steps, the learning rate (and weight decay) is reduced.

    Parameters
    ----------
        optimizer: Optimizer, list:
            Wrapped optimizer.
        scheduler: LRSchedule, list
            Learning rate schedule of the optimizer.
            Asserts that the second schedule belongs to second optimizer and so on.
        mode: str
            One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor: float
            Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience: int
            Number of steps with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 steps
            with no improvement, and will only decrease the LR after the
            3rd step if the loss still hasn't improved then.
            Default: 10.
        threshold: float
            Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        max_reduce: int
            Number of maximum decays on plateaus. Default: 10.
        threshold_mode: str
            One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown: int
            Number of steps to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        eps: float
            Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose: bool
            If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        scheduler,
        factor=0.1,
        patience=10,
        threshold=1e-4,
        max_reduce=10,
        cooldown=0,
        threshold_mode="rel",
        min_lr=0,
        eps=1e-8,
        mode="min",
        verbose=False,
    ):

        if factor >= 1.0:
            raise ValueError(f"Factor should be < 1.0 but is {factor}.")
        self.factor = factor
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(optimizer, MultiWrapper):
            self.optimizer = optimizer.wrapped
        if isinstance(scheduler, MultiWrapper):
            self.scheduler = scheduler.wrapped

        if not isinstance(self.optimizer, (list,tuple)):
            self.optimizer = [self.optimizer]
        if not isinstance(self.scheduler, (list,tuple)):
            self.scheduler = [self.scheduler]

        assert len(self.optimizer) == len(self.scheduler)

        for opt in self.optimizer:
            # Attach optimizer
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(f"{type(opt).__name__} is not an Optimizer but is of type {type(opt)}")

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_steps = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_step = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()
        self._reduce_counter = 0

    def _reset(self):
        """Resets num_bad_steps counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_steps = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        step = self.last_step + 1
        self.last_step = step

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_steps = 0  # ignore any bad steps in cooldown

        if self.num_bad_steps > self.patience:
            self._reduce(step)
            self.cooldown_counter = self.cooldown
            self.num_bad_steps = 0

    def _reduce(self, step):
        self._reduce_counter += 1

        for optimzer, schedule in zip(self.optimizer, self.scheduler):
            if hasattr(schedule, "base_lrs"):
                schedule.base_lrs = [lr * self.factor for lr in schedule.base_lrs]
            else:
                raise ValueError(
                    "Schedule does not have attribute 'base_lrs' for the learning rate."
                )
        if self.verbose:
            logging.info(f"Step {step}: reducing on plateu by {self.factor}.")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = np.inf
        else:  # mode == 'max':
            self.mode_worse = -np.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "scheduler"]
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )