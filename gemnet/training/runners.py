"""
Runner class handles the whole training/validation/testing procedure
and the model saving
"""
import os, sys
import logging
import numpy as np
import yaml
import string
import ast
import random
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import copy

from gemnet.model import gemnet 
from gemnet.training import trainers
from gemnet.training.metrics import Metrics, BestMetrics, spearmanr
import gemnet.training.data_container as data_containers
from gemnet.training.data_provider import DataProvider
from easydict import EasyDict
import torch
import pprint
from gemnet.utils.config_utils import update_config, dump_config
from gemnet.utils import dist_utils

class BaseRunner(object):
    """
    An epoch based Runner class for the whole training procedure
    """
    def __init__(self, args, other_args) -> None:
        self.args = args
        self.other_args = other_args
        self.rank = dist_utils.get_rank()

        self.init_logger()
        self.init_config()
        self.init_logdir()
        self.init_model()
        self.init_data()
        self.load_pretrained()
        self.init_trainer()
        self.init_metrics()
        self.resume()
        logging.info("Runner initialization done!")

    def build_dataset(self, config):
        class_name = config["dataset_class"]
        return getattr(data_containers, class_name).from_config(config)

    def init_logger(self):

        self.logger = logging.getLogger()
        self.logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel("INFO")

    def init_config(self):
        with open(self.args.config, 'r') as c:
            self.config = yaml.safe_load(c)

        for key, val in self.config.items():
            if type(val) is str:
                try:
                    self.config[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    pass

        self.config = EasyDict(self.config)
        update_config(self.config, self.other_args)
        self.config.model["num_targets"] = 2 if self.config.model.mve else 1
        torch.manual_seed(self.config.tfseed)
        config_str = pprint.pformat(self.config)
        if self.rank== 0:
            logging.info("config:\n {}".format(config_str))

    def init_logdir(self):

        logging.info("Start training")
        num_gpus = torch.cuda.device_count()
        cuda_available = torch.cuda.is_available()
        logging.info(f"Available GPUs: {num_gpus}")
        logging.info(f"CUDA Available: {cuda_available}")
        if num_gpus == 0:
            logging.warning("No GPUs were found. Training is run on CPU!")
        if not cuda_available:
            logging.warning("CUDA unavailable. Training is run on CPU!")

        if (self.config.restart is None) or (self.config.restart == "None"): 
            self.directory = os.path.join(self.config.logdir,  os.path.basename(self.args.config), datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            self.directory = self.config.restart

        ## set logger

        logging.info(f"Directory: {self.directory}")
        logging.info("Create directories")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)

        self.best_dir = os.path.join(self.directory, "best")
        if not os.path.exists(self.best_dir):
            os.makedirs(self.best_dir, exist_ok=True)
        self.log_dir = os.path.join(self.directory, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        fh = logging.FileHandler(os.path.join(self.directory, "log.txt"))
        self.logger.addHandler(fh)    

        extension = ".pth"
        self.log_path_model = f"{self.log_dir}/model{extension}"
        self.log_path_training = f"{self.log_dir}/training{extension}"
        self.best_path_model = f"{self.best_dir}/model{extension}"
        self.config_save_path = os.path.join(self.directory, os.path.basename(self.args.config))
        dump_config(dict(self.config), self.config_save_path)

    def init_model(self):
        logging.info("Initialize model")
        if "class" in self.config.model:
            model_class = self.config.model.get("class")
            self.config.model.pop("class")
        else:
            model_class = "GemNet"
        self.model = getattr(gemnet, model_class)(**self.config.model)

    def init_data(self):
        logging.info("Building dataset")
        self.data_container = self.build_dataset(self.config)

        logging.info(f"Total dataset length: {len(self.data_container)}")

        self.data_provider = DataProvider(self.data_container, self.config.num_train, 
                                          self.config.num_val, self.config.batch_size, 
                                          seed=self.config.data_seed, shuffle=True, 
                                          random_split=False)

    def load_pretrained(self):
        logging.info("Prepare training")

        ## Load from pretrained
        if "pretrained" in self.config.keys():
            if os.path.exists(self.config.pretrained):
                logging.info(f"load pretrained model from {self.config.pretrained}")
                model_checkpoint = torch.load(self.config.pretrained, map_location="cpu")['model']
                msg = self.model.load_state_dict(model_checkpoint, strict=False)
                logging.info(f"load message: {msg}")

    def init_trainer(self):
        # Initialize trainer
        trainer_class = self.config.trainer.pop("class")
        self.trainer = getattr(trainers, trainer_class)(model=self.model, **self.config.trainer)
        self.device = self.trainer.device

    def init_metrics(self):
        # Initialize metrics
        self.train_metrics = Metrics("train", self.trainer.tracked_metrics)
        self.val_metrics = Metrics("val", self.trainer.tracked_metrics)
        self.test_metrics = Metrics("test", self.trainer.tracked_metrics)


        # Save/load best recorded loss (only the best model is saved)
        self.metrics_best_val = BestMetrics(self.best_dir, self.val_metrics, 
                                main_metric=self.config.main_metric, 
                                metric_mode=self.config.metric_mode)
        self.metrics_best_test = BestMetrics(self.best_dir, self.test_metrics, 
                                main_metric=self.config.main_metric,
                                metric_mode=self.config.metric_mode)
    
    def resume(self):
        if os.path.exists(self.config.get("resume", "")):
            resume_dir = os.path.dirname(self.config.get("resume", ""))
            resume_epoch = self.config.get("resume_epoch", None)
            if resume_epoch is None:
                model_path = os.path.join(resume_dir, "model.pth")
                training_path = os.path.join(resume_dir, "training.pth")
                resume_epoch = 0
            else:
                model_path = os.path.join(resume_dir, "model_{}.pth".format(resume_epoch))
                training_path = os.path.join(resume_dir, "training_{}.pth".format(resume_epoch))

            logging.info("Resume model and trainer from {}, epoch is {}".format(resume_dir, resume_epoch))
    
            model_checkpoint = torch.load(model_path, map_location=self.trainer.device)
            self.model.load_state_dict(model_checkpoint["model"])

            train_checkpoint = torch.load(training_path, map_location=self.trainer.device)
            self.trainer.load_state_dict(train_checkpoint["trainer"])
            # restore the best saved results
            self.metrics_best_val.inititalize()
            self.start_epoch = resume_epoch
        else:
            logging.info("Freshly initialize model")
            self.metrics_best_val.inititalize()
            self.start_epoch = 0

    def run(self):
        raise NotImplementedError()


class DownStreamRunner(BaseRunner):
    """
    downstream task runner with train/val/test, single dataset.
    """
    
    def run(self):
        best_epoch = -1

        for epoch in tqdm(range(self.start_epoch, self.config.num_epochs)):
            # Perform training step
            self.trainer.train_on_epoch(self.data_provider, self.train_metrics, self.config.iter_per_epoch)
            
            # Save progress
            if epoch % self.config.save_interval == 0 and self.trainer.rank==0:
                torch.save({"model": self.model.state_dict()}, self.log_path_model)
                torch.save(
                    {"trainer": self.trainer.state_dict(), "step": epoch}, self.log_path_training
                )

            # Check performance on the validation set
            if epoch % self.config.evaluation_interval == 0:
                # Save backup variables and load averaged variables
                self.trainer.save_variable_backups()
                self.trainer.load_averaged_variables()

                # Evaluation
                self.trainer.eval_on_epoch(self.data_provider, self.val_metrics, split="val")
                self.trainer.eval_on_epoch(self.data_provider, self.test_metrics, split="test")

                # Update and save best result <this is very trainer specific actually
                if self.trainer.rank==0:
                    if self.metrics_best_val.is_best(self.val_metrics):
                        best_epoch = epoch
                        last_best =  self.metrics_best_val.main_metric
                        self.metrics_best_val.update(epoch, self.val_metrics)
                        self.metrics_best_test.update(epoch, self.test_metrics)
                        logging.info(f"best {self.metrics_best_val.main_metric_name} on valid update: {last_best} => {self.metrics_best_val.main_metric}")
                        logging.info(f"current spearman rho on test: {self.metrics_best_test.main_metric}")
                        torch.save(self.model.state_dict(), self.best_path_model)
                    else:
                        logging.info(f"best {self.metrics_best_val.main_metric_name} on valid unchanged, best epoch: {best_epoch} , value: {self.metrics_best_val.main_metric}")
                        logging.info(f"best valid model {self.metrics_best_test.main_metric_name} on test: {self.metrics_best_test.main_metric}")
                train_metrics_res = self.train_metrics.result(append_tag=False)
                val_metrics_res = self.val_metrics.result(append_tag=False)
                test_metrics_res = self.test_metrics.result(append_tag=False)
                metrics_strings = [
                    f"{key}: train={train_metrics_res[key]:.6f}, val={val_metrics_res[key]:.6f}, test={test_metrics_res[key]:.6f} \n"
                    for key in self.val_metrics.keys
                ]
                if self.trainer.rank==0:
                    logging.info(
                        f"epoch ({epoch}): " + "; ".join(metrics_strings)
                    )

                # decay learning rate on plateau
                self.trainer.decay_maybe(self.val_metrics.loss)

                self.train_metrics.reset_states()
                self.val_metrics.reset_states()
                self.test_metrics.reset_states()

                # Restore backup variables
                self.trainer.restore_variable_backups()

                # early stopping
                if self.trainer.rank==0 and self.config.early_stop:
                    if epoch - self.metrics_best_val.step > self.config.patience * self.config.evaluation_interval:
                        logging.info("early stoped.")

                        result = {key + "valid_best": val for key, val in self.metrics_best_val.items()}
                        result_test = {key + "test": val for key, val in self.metrics_best_test.items()}
                        result.update(result_test)
                        for key, val in result.items():
                            logging.info(f"{key}: {val}")
                            sys.exit(0)


class PretrainRunner(BaseRunner):
    """
    pretrain task runner with only training, maybe multiple datasets.
    """

    def init_data(self):
        # no need for data initialization.
        pass

    def run(self):
        for epoch in tqdm(range(self.start_epoch, self.config.num_epochs)):
            # prepare data, support re
            all_data_path = list()
            if isinstance(self.config.dataset, str):
                self.config.dataset = [self.config.dataset]
            for path_str in self.config.dataset:
                dir_name = os.path.dirname(path_str)
                file_name = os.path.basename(path_str)
                dir_path_obj = Path(dir_name)
                matched_files = list(dir_path_obj.rglob(file_name))
                matched_files = [str(x) for x in matched_files if os.path.exists(str(x))]
                all_data_path.extend(matched_files)
            all_data_path = sorted(list(set(all_data_path)))

            for data_path in all_data_path:
                ## process re
                new_config = copy.deepcopy(self.config)
                new_config["dataset"] = data_path

                data_container = self.build_dataset(new_config)

                logging.info(f"data path {new_config.dataset} dataset length: {len(data_container)}")

                data_provider = DataProvider(data_container, len(data_container), 0,
                                            self.config.batch_size, seed=self.config.data_seed, 
                                            shuffle=True, 
                                            random_split=False)
                # Perform training step
                self.trainer.train_on_epoch(data_provider, self.train_metrics, self.config.iter_per_epoch)
                # Save progress
                if self.trainer.rank==0:
                    logging.info("saving latest models after training on data {}".format(new_config.dataset))
                    torch.save({"model": self.model.state_dict()}, self.log_path_model)
                    torch.save(
                        {"trainer": self.trainer.state_dict(), "step": epoch}, self.log_path_training
                    )

            # Save progress: non latest
            if epoch % self.config.save_interval == 0 and self.trainer.rank==0:
                logging.info("saving models for epoch {}".format(epoch))
                model_save_path = os.path.join(os.path.dirname(self.log_path_model), "model_{}.pth".format(epoch))
                trainer_save_path = os.path.join(os.path.dirname(self.log_path_training), "training_{}.pth".format(epoch))
                torch.save({"model": self.model.state_dict()}, model_save_path)
                torch.save(
                    {"trainer": self.trainer.state_dict(), "step": epoch}, trainer_save_path
                )

                # Save backup variables and load averaged variables
                self.trainer.save_variable_backups()
                self.trainer.load_averaged_variables()

                train_metrics_res = self.train_metrics.result(append_tag=False)
                metrics_strings = [f"{key}: train={train_metrics_res[key]:.6f}" for key in self.train_metrics.keys]
                if self.trainer.rank==0:
                    logging.info(
                        f"epoch ({epoch}): " + "; ".join(metrics_strings)
                    )

            # decay learning rate on plateau
            self.trainer.decay_maybe(self.train_metrics.loss)

            self.train_metrics.reset_states()

            # Restore backup variables
            self.trainer.restore_variable_backups()