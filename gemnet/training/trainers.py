import logging
import torch
from torch.functional import broadcast_shapes
from .schedules import LinearWarmupExponentialDecay, ReduceLROnPlateau, MultiWrapper
from .ema_decay import ExponentialMovingAverage
from gemnet.utils import dist_utils, training_utils
from gemnet.training import metrics
import torch.nn as nn
from itertools import islice
import torch.distributed as dist
from tqdm import tqdm

class BaseTrainer(object):
    """
    GYF: a base trainer class. We add support for distributed training
    """

    def forwrad_and_backward(self, model, metrics, inputs, targets):
        raise NotImplementedError

    def init_distributed(self):
        self.rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()

        dist_utils.check_and_initialize(self.world_size, self.rank)
        logging.info("Distributed Initialized")
        self.device = dist_utils.get_device(self.world_size, self.rank)
        self.model = self.model.to(self.device)
        if self.world_size > 1:
            logging.info("Converting sync batch norm")
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def train_on_epoch(self, data_provider, metrics, iter_per_epoch=None):
        """
        GYF: we support distributed training in this function.
        """
        if self.world_size > 1 and dist.is_initialized():
            data_loader = data_provider.get_distributed_loader("train", self.world_size, self.rank)
        else:
            data_loader = data_provider.get_loader("train")

        iter_per_epoch = len(data_loader) if iter_per_epoch is None else iter_per_epoch
        model = self.model
        if self.world_size > 1 and dist.is_initialized():
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device], 
                                                find_unused_parameters=True)
        model.train()
        for idx, batch in enumerate(islice(data_loader, iter_per_epoch)):
            inputs, targets = batch
            self.forwrad_and_backward(model, metrics, inputs, targets)
            if idx % getattr(self, "log_iter", 20) == 0 and self.rank==0:
                result = metrics.result(append_tag=False)
                metrics_strings = [
                f"{key}: {result[key]:.4f}"
                for key in metrics.keys]
                metrics_strings = " ;".join(metrics_strings)  
                logging.info("iter {} | train metrics: {}".format(idx, metrics_strings))

    def dict2device(self, data, device=None):
        if device is None:
            device = self.device
        for key in data:
            data[key] = data[key].to(device)
        return data 

class Trainer(BaseTrainer):
    """
    Parameters
    ----------
        model: Model
            Model to train.
        learning_rate: float
            Initial learning rate.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule..
        weight_decay: bool
            Weight decay factor of the AdamW optimizer.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay
        grad_clip_max: float
            Gradient clipping threshold.
        decay_patience: int
            Learning rate decay on plateau. Number of evaluation intervals after decaying the learning rate.
        decay_factor: float
            Learning rate decay on plateau. Multiply inverse of decay factor by learning rate to obtain new learning rate.
        decay_cooldown: int
            Learning rate decay on plateau. Number of evaluation intervals after which to return to normal operation.
        ema_decay: float
            Decay to use to maintain the moving averages of trained variables.
        rho_force: float
            Weighing factor for the force loss compared to the energy. In range [0,1]
            loss = loss_energy * (1-rho_force) + loss_force * rho_force
        loss: str
            Name of the loss objective of the forces.
        mve: bool
            If True perform Mean Variance Estimation.
        agc: bool
            If True use adaptive gradient clipping else clip by global norm.
        finetune_mode: str
            The mode to finetune a pre-trained GemNet EBM (None, `tune_energy_map`, `tune_output_block`, `tune_all`).
        lr_ratio: float
            The learning ratio of interaction blocks against output blocks.

    Fine-tuning strategies:
    1. Fix the whole GemNet (interaction blocks and output blocks), only tune a linear mapping of energy.
        Implementation: load the pre-trained model, set `finetune_mode` to `tune_energy_map`,
                        fix the parameters of all other blocks.
    2. Train new output blocks upon the representations extracted by GemNet, fix the parameters of interaction blocks.
        Implementation: load the pre-trained model, set `finetune_mode` to `tune_output_block`,
                        randomly initialize the parameters of output blocks, fix the parameters of other modules.
    3. Fine-tune the whole GemNet model.
        Implementation: load the pre-trained model, set `finetune_mode` to `tune_all`,
                        randomly initialize the parameters of output blocks,
                        set `lr_ratio` to a value less 1 (e.g. 0.1 by default) to fine-tune the encoder.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        decay_steps: int = 100000,
        decay_rate: float = 0.96,
        warmup_steps: int = 0,
        weight_decay: float = 0.001,
        staircase: bool = False,
        grad_clip_max: float = 1000,
        decay_patience: int = 10,  # decay lr on plateau by decay_factor
        decay_factor: float = 0.5,
        decay_cooldown: int = 10,
        ema_decay: float = 0.999,
        rho_force: float = 0.99,
        loss: str = "mae",  # else use rmse
        mve: bool = False,
        agc=False,
        finetune_mode=None,
        lr_ratio=0.1,
        log_iter=20,
    ):
        assert 0 <= rho_force <= 1
        self.model = model
        self.ema_decay = ema_decay
        self.grad_clip_max = grad_clip_max
        self.rho_force = float(rho_force)
        self.mve = mve
        self.loss = loss
        self.agc = agc
        self.finetune_mode = finetune_mode
        self.lr_ratio = lr_ratio
        self.log_iter = log_iter
        self.init_distributed()

        self.reset_optimizer(
            learning_rate,
            weight_decay,
            warmup_steps,
            decay_steps,
            decay_rate,
            staircase,
            decay_patience,
            decay_factor,
            decay_cooldown,
        )

    @property
    def tracked_metrics(self):
        if self.mve:
            return ["loss","energy_mae","energy_nll","energy_var", 
                    "force_mae", "force_rmse", "force_nll", "force_var"]
        else:
            return ["loss", "energy_mae", "force_mae", "force_rmse"]

    def reset_optimizer(
        self,
        learning_rate,
        weight_decay,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase,
        decay_patience,
        decay_factor,
        decay_cooldown,
    ):
        if weight_decay > 0:
            adamW_params = []
            rest_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "atom_emb" in name:
                        rest_params += [param]
                        continue
                    if "frequencies" in name:
                        rest_params += [param]
                        continue
                    if "bias" in name:
                        rest_params += [param]
                        continue
                    adamW_params += [param]

            if self.finetune_mode == "tune_all":
                all_params = self.model.parameters()
                output_params = self.model.out_blocks.parameters()
                encoder_params = list(set(all_params) - set(output_params))
                adamW_params_new = []
                adamW_params_new.append({"params": list(set(adamW_params).intersection(set(encoder_params))),
                                         "lr": self.lr_ratio * learning_rate})
                adamW_params_new.append({"params": list(set(adamW_params).intersection(set(output_params))),
                                         "lr": learning_rate})
                adamW_params = adamW_params_new
                rest_params_new = []
                rest_params_new.append({"params": list(set(rest_params).intersection(set(encoder_params))),
                                        "lr": self.lr_ratio * learning_rate})
                rest_params_new.append({"params": list(set(rest_params).intersection(set(output_params))),
                                        "lr": learning_rate})
                rest_params = rest_params_new

            # AdamW optimizer
            AdamW = torch.optim.AdamW(
                adamW_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                weight_decay=weight_decay,
                amsgrad=True,
            )
            lr_schedule_AdamW = LinearWarmupExponentialDecay(
                AdamW, warmup_steps, decay_steps, decay_rate, staircase
            )

            # Adam: Optimzer for embeddings, frequencies and biases
            Adam = torch.optim.Adam(
                rest_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                amsgrad=True,
            )
            lr_schedule_Adam = LinearWarmupExponentialDecay(
                Adam, warmup_steps, decay_steps, decay_rate, staircase
            )

            # Wrap multiple optimizers to ease optimizer calls later on
            self.schedulers = MultiWrapper(
                lr_schedule_AdamW, lr_schedule_Adam
            )
            self.optimizers = MultiWrapper(AdamW, Adam)

        else:
            if self.finetune_mode == "tune_all":
                train_parameters = []
                all_params = self.model.parameters()
                output_params = self.model.out_blocks.parameters()
                encoder_params = list(set(all_params) - set(output_params))
                train_parameters.append({"params": encoder_params, "lr": self.lr_ratio * learning_rate})
                train_parameters.append({"params": output_params, "lr": learning_rate})
            else:
                train_parameters = self.model.parameters()

            # Adam: Optimzer for all parameters
            Adam = torch.optim.Adam(
                train_parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                amsgrad=True,
            )
            lr_schedule_Adam = LinearWarmupExponentialDecay(
                Adam, warmup_steps, decay_steps, decay_rate, staircase
            )

            # Also wrap single optimizer for unified interface later
            self.schedulers = MultiWrapper(lr_schedule_Adam)
            self.optimizers = MultiWrapper(Adam)

        # Learning rate decay on plateau
        self.plateau_callback = ReduceLROnPlateau(
            optimizer=self.optimizers,
            scheduler=self.schedulers,
            factor=decay_factor,
            patience=decay_patience,
            cooldown=decay_cooldown,
            verbose=True,
        )

        if self.agc:
            # adaptive gradient clipping should not modify the last layer (see paper)
            self.params_except_last = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "out_energy" in name:
                        self.params_except_last += [param]
                    if "out_forces" in name:
                        self.params_except_last += [param]

        if self.finetune_mode in ["tune_energy_map", "tune_output_block"]:
            all_params = self.model.parameters()
            tuning_params = self.model.energy_map_blocks.parameters() if self.finetune_mode == "tune_energy_map" \
                else self.model.out_blocks.parameters()
            fix_params = list(set(all_params) - set(tuning_params))
            for p in fix_params:
                p.requires_grad = False
            if self.finetune_mode == "tune_energy_map":
                self.model.energy_map_blocks.apply(self._init_weights)
            else:
                self.model.out_blocks.apply(self._init_weights)
        elif self.finetune_mode == "tune_all":
            self.model.out_blocks.apply(self._init_weights)

        self.exp_decay = ExponentialMovingAverage(
            [p for p in self.model.parameters() if p.requires_grad], self.ema_decay
        )

    def _init_weights(self, module, initializer_range=0.02):
        """
        Initialize the weights of each module.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_variable_backups(self):
        self.exp_decay.store()

    def load_averaged_variables(self):
        self.exp_decay.copy_to()

    def restore_variable_backups(self):
        self.exp_decay.restore()

    def decay_maybe(self, val_loss):
        self.plateau_callback.step(val_loss)

    def get_mae(self, targets, pred):
        """
        Mean Absolute Error
        """
        return torch.nn.functional.l1_loss(pred, targets, reduction="mean")

    def get_rmse(self, targets, pred):
        """
        Root Mean Squared Error
        """
        return torch.sqrt(torch.nn.functional.mse_loss(pred, targets, reduction="mean"))

    def get_nll(self, targets, mean_pred, var_pred):
        return torch.nn.functional.gaussian_nll_loss(
            mean_pred, targets, var_pred, reduction="mean"
        )

    def predict(self, inputs, model=None):
        if model is None:
            model = self.model
        energy, forces = model(inputs)

        if self.mve:
            mean_energy = energy[:, :1]
            var_energy = torch.nn.functional.softplus(energy[:, 1:])
            mean_forces = forces[:, 0, :]
            var_forces = torch.nn.functional.softplus(forces[:, 1, :])
            return mean_energy, var_energy, mean_forces, var_forces
        else:
            if len(forces.shape) == 3:
                forces = forces[:, 0]
            return energy, None, forces, None

    def predict_on_batch(self, dataset_iter):
        inputs, _ = next(dataset_iter)
        inputs = self.dict2device(inputs)
        return self.predict(inputs)
            
    def forwrad_and_backward(self, model, metrics, inputs, targets):
        inputs, targets = self.dict2device(inputs), self.dict2device(targets)
        mean_energy, var_energy, mean_forces, var_forces = self.predict(inputs, model)
        if self.mve:
            energy_nll = self.get_nll(targets["E"], mean_energy, var_energy)
            force_nll = self.get_nll(targets["F"], mean_forces, var_forces)
            loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll
        else:
            energy_mae = self.get_mae(targets["E"], mean_energy)
            if self.loss == "mae":
                force_metric = self.get_mae(targets["F"], mean_forces)
            else:
                force_metric = self.get_rmse(targets["F"], mean_forces)
            loss = energy_mae * (1 - self.rho_force) + self.rho_force * force_metric
        self.optimizers.zero_grad()
        loss.backward()
        self.model.scale_shared_grads()

        if self.agc:
            training_utils.adaptive_gradient_clipping(
                self.params_except_last, clip_factor=self.grad_clip_max
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_max
            )

        self.optimizers.step()
        self.schedulers.step()
        self.exp_decay.update()

        loss = loss.detach()
        with torch.no_grad():
            if self.mve:
                energy_mae = self.get_mae(targets["E"], mean_energy)
                force_mae = self.get_mae(targets["F"], mean_forces)
                force_rmse = self.get_rmse(targets["F"], mean_forces)

            else:
                if self.loss == "mae":
                    force_mae = force_metric
                    force_rmse = self.get_rmse(targets["F"], mean_forces)
                else:
                    force_mae = self.get_mae(targets["F"], mean_forces)
                    force_rmse = force_metric

            if self.mve:
                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                    energy_nll=energy_nll,
                    energy_var=var_energy,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                    force_nll=force_nll,
                    force_var=var_forces,
                )
            else:
                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                )

        return loss

    def train_on_batch(self, dataset_iter, metrics):
        """
        GYF: this function only support single gpu training because 
        self.model is not wrapped by DistributedDataParallel; to be depricated.

        """
        self.model.train()
        inputs, targets = next(dataset_iter)
        # push to GPU if available
        loss = self.forwrad_and_backward(self.model, metrics, inputs, targets)
        return loss

    def test_on_batch(self, dataset_iter, metrics):
        self.model.eval()
        inputs, targets = next(dataset_iter)
        # push to GPU if available
        inputs, targets = self.dict2device(inputs), self.dict2device(targets)

        if self.model.direct_forces:
            # do not need any gradients -> reduce memory consumption
            with torch.no_grad():
                mean_energy, var_energy, mean_forces, var_forces = self.predict(inputs)
        else:
            # need gradient for forces
            mean_energy, var_energy, mean_forces, var_forces = self.predict(inputs)

        with torch.no_grad():
            energy_mae = self.get_mae(targets["E"], mean_energy)
            force_mae = self.get_mae(targets["F"], mean_forces)
            force_rmse = self.get_rmse(targets["F"], mean_forces)

            if self.mve:
                energy_nll = self.get_nll(targets["E"], mean_energy, var_energy)
                loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_mae
                force_nll = self.get_nll(targets["F"], mean_forces, var_forces)
                loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll

                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                    energy_nll=energy_nll,
                    energy_var=var_energy,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                    force_nll=force_nll,
                    force_var=var_forces,
                )

            else:
                if self.loss == "mae":
                    force_metric = self.get_mae(targets["F"], mean_forces)
                else:
                    force_metric = self.get_rmse(targets["F"], mean_forces)

                loss = energy_mae * (1 - self.rho_force) + self.rho_force * force_metric

                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                )

        return loss

    def eval_on_batch(self, dataset_iter):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = next(dataset_iter)
            # push to GPU if available
            inputs, targets = self.dict2device(inputs), self.dict2device(targets)
            energy, _, forces, _ = self.predict(inputs)
        return (energy, forces), targets

    def state_dict(self):
        """Returns the state of the trainer and all subinstancces except the model."""
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in [
                "model",
                "schedulers",
                "optimizers",
                "plateau_callback",
                "exp_decay",
            ]
        }
        for attr in ["schedulers", "optimizers", "plateau_callback", "exp_decay"]:
            state_dict.update({attr: getattr(self, attr).state_dict()})
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        trainer_dict = {
            key: value
            for key, value in self.state_dict().items()
            if key
            not in [
                "model",
                "schedulers",
                "optimizers",
                "plateau_callback",
                "exp_decay",
            ]
        }
        self.__dict__.update(trainer_dict)
        for attr in ["schedulers", "optimizers", "plateau_callback", "exp_decay"]:
            getattr(self, attr).load_state_dict(state_dict[attr])


class DDGTrainer(Trainer):
    """ Trainer for predicting ddG change between 
    wild type and mutants. We use the difference between 
    energy to measure the ddG change.

    """
    @property
    def tracked_metrics(self):
        return ["loss", "energy_mae", "force_mae", "force_rmse", "spearman"]

    def predict_on_batch(self, dataset_iter):
        inputs, _ = next(dataset_iter)
        outputs = []
        for input_dict in inputs:
            input_dict = self.dict2device(input_dict)
            results = self.predict(input_dict)
            outputs.append(results)
        return outputs

    def forwrad_and_backward(self, model, metrics, inputs, targets):
        inputs_wt, inputs_mt = inputs
        targets, targets_1 = targets
        # push to GPU if available
        inputs_wt, inputs_mt, targets, targets_1 = self.dict2device(inputs_wt), self.dict2device(inputs_mt), self.dict2device(targets), self.dict2device(targets_1)
        mean_energy_wt, var_energy_wt, mean_forces_wt, var_forces_wt = self.predict(inputs_wt, model)
        mean_energy_mt, var_energy_mt, mean_forces_mt, var_forces_mt = self.predict(inputs_mt, model)
        mean_energy = mean_energy_mt - mean_energy_wt
        var_energy = var_energy_mt
        energy_mae = self.get_mae(targets["E"], mean_energy)
        force_loss = self.get_mae(targets["F"], mean_forces_wt)
        force_loss1 = self.get_mae(targets_1["F"], mean_forces_mt)
        loss = energy_mae + 0.0 * force_loss + 0.0 * force_loss1

        self.optimizers.zero_grad()
        loss.backward()
        self.model.scale_shared_grads()

        if self.agc:
            training_utils.adaptive_gradient_clipping(
                self.params_except_last, clip_factor=self.grad_clip_max
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_max
            )

        self.optimizers.step()
        self.schedulers.step()
        self.exp_decay.update()

        # no gradients needed anymore
        loss = loss.detach()
        with torch.no_grad():
            # update molecule metrics
            metrics.update_state(
                nsamples=mean_energy.shape[0],
                loss=loss,
                energy_mae=energy_mae,
            )

        return loss

    def test_on_batch(self, dataset_iter, metrics):
        """
        GYF: again, no force, energy as ddG
        """
        self.model.eval()
        inputs, targets = next(dataset_iter)
        # push to GPU if available
        inputs, targets = next(dataset_iter)
        inputs_wt, inputs_mt = inputs
        targets = targets[0]
        # push to GPU if available
        inputs_wt, inputs_mt, targets = self.dict2device(inputs_wt), self.dict2device(inputs_mt), self.dict2device(targets)

        with torch.no_grad():
            mean_energy_wt, var_energy_wt, mean_forces_wt, var_forces_wt = self.predict(inputs_wt)
            mean_energy_mt, var_energy_mt, mean_forces_mt, var_forces_mt = self.predict(inputs_mt)
            mean_energy = mean_energy_mt - mean_energy_wt
            var_energy = var_energy_mt

            energy_mae = self.get_mae(targets["E"], mean_energy)

            if self.mve:
                energy_nll = self.get_nll(targets["E"], mean_energy, var_energy)

                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    energy_mae=energy_mae,
                    energy_nll=energy_nll,
                    energy_var=var_energy,
                )

            else:
                loss = energy_mae

                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                )

        return loss
    
    @torch.no_grad()
    def eval_on_batch(self, dataset_iter):
        self.model.eval()
        inputs, targets = next(dataset_iter)
        # push to GPU if available
        inputs_wt, inputs_mt = inputs
        targets = targets[0]
        # push to GPU if available
        inputs_wt, inputs_mt, targets = self.dict2device(inputs_wt), self.dict2device(inputs_mt), self.dict2device(targets)
        mean_energy_wt, var_energy_wt, mean_forces_wt, var_forces_wt = self.predict(inputs_wt)
        mean_energy_mt, var_energy_mt, mean_forces_mt, var_forces_mt = self.predict(inputs_mt)
        mean_energy = mean_energy_mt - mean_energy_wt
        var_energy = var_energy_mt
        return mean_energy, targets

    @torch.no_grad()
    def pred_on_batch(self, batch):
        inputs, targets = batch
        wild_type, mutant = inputs
        targets = targets[0]
        wild_type, mutant, targets = self.dict2device(wild_type), self.dict2device(mutant), self.dict2device(targets)
        mean_energy_wt, var_energy_wt, mean_forces_wt, var_forces_wt = self.predict(wild_type)
        mean_energy_mt, var_energy_mt, mean_forces_mt, var_forces_mt = self.predict(mutant)
        mean_energy = mean_energy_mt - mean_energy_wt
        var_energy = var_energy_mt
        return mean_energy, targets

    def eval_on_epoch(self, data_provider, metrics, split="val"):
        data_loader = data_provider.get_loader(split)
        self.model.eval()
        with torch.no_grad():
            all_energies = list()
            all_targets = list()
            for batch in tqdm(data_loader):
                inputs, targets = batch
                energy, targets = self.pred_on_batch(batch)
                
                all_energies.append(energy)
                all_targets.append(targets["E"])
        
        results = self.evaluate(torch.cat(all_energies).view(-1), torch.cat(all_targets).view(-1))
        metrics.update_state(**results)
    
    def evaluate(self, pred, targets):
        metric_dict = dict()
        rho = metrics.spearmanr(pred, targets)
        metric_dict["spearman"] = rho
        metric_dict["nsamples"] = pred.shape[0]
        return metric_dict

class EBMTrainer(Trainer):
    """
    Trainer for energy-based model.
    We contrast between positive side chain conformations with negative side chain conformations.
    """

    def __init__(
            self,
            model,
            learning_rate: float = 1e-3,
            decay_steps: int = 100000,
            decay_rate: float = 0.96,
            warmup_steps: int = 0,
            weight_decay: float = 0.001,
            staircase: bool = False,
            grad_clip_max: float = 1000,
            decay_patience: int = 10,  # decay lr on plateau by decay_factor
            decay_factor: float = 0.5,
            decay_cooldown: int = 10,
            ema_decay: float = 0.999,
            rho_force: float = 0.99,
            loss: str = "mae",  # else use rmse
            mve: bool = False,
            agc=False,
            num_negative: int = 1,
            log_iter: int =20,
    ):
        super(EBMTrainer, self).__init__(model, learning_rate, decay_steps, decay_rate, warmup_steps, weight_decay,
                                         staircase, grad_clip_max, decay_patience, decay_factor, decay_cooldown,
                                         ema_decay, rho_force, loss, mve, agc, log_iter=log_iter)
        self.num_negative = num_negative

    @property
    def tracked_metrics(self):
        return ["loss", "energy_mae", "force_mae", "force_rmse", "log_likelihood"]

    def get_log_likelihood(self, all_energy, eps=1e-6):
        all_energy = all_energy.view(-1).view(self.num_negative + 1, -1).permute(1, 0).clamp(-1e2, 1e2)  # (B, N_neg + 1)
        labels = torch.zeros(all_energy.shape[0], dtype=torch.long, device=all_energy.device)
        return -nn.functional.cross_entropy(all_energy, labels)

    def forwrad_and_backward(self, model, metrics, inputs, targets):
        inputs, targets = self.dict2device(inputs), self.dict2device(targets)
        mean_energy, var_energy, mean_forces, var_forces = self.predict(inputs, model)
        energy_mae = self.get_mae(targets["E"], mean_energy)
        force_mae = self.get_mae(targets["F"], mean_forces)
        log_likelihood = self.get_log_likelihood(mean_energy)
        loss = -log_likelihood + 0.0 * energy_mae + 0.0 * force_mae

        self.optimizers.zero_grad()
        loss.backward()
        self.model.scale_shared_grads()

        if self.agc:
            training_utils.adaptive_gradient_clipping(
                self.params_except_last, clip_factor=self.grad_clip_max
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_max
            )

        self.optimizers.step()
        self.schedulers.step()
        self.exp_decay.update()

        # no gradients needed anymore
        loss = loss.detach()
        log_likelihood = log_likelihood.detach()
        with torch.no_grad():
            # update molecule metrics
            metrics.update_state(
                nsamples=mean_energy.shape[0] // (self.num_negative + 1),
                loss=loss,
                log_likelihood=log_likelihood
            )

        return loss

    def eval_on_epoch(self, data_provider, metrics, split="val"):
        # pretrained trainer without evaluation
        return metrics
