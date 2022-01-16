"""
Training with multiple data
"""
import argparse
import os
import logging
import numpy as np
import yaml
import string
import ast
import random
import time
from datetime import datetime
from tqdm import tqdm
import copy
from pathlib import Path

from gemnet.model.gemnet import GemNet
from gemnet.training import trainers
from gemnet.training.metrics import Metrics, BestMetrics, spearmanr
import gemnet.training.data_container as data_containers
from gemnet.training.data_provider import DataProvider
from easydict import EasyDict
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

def parse_args():
    parser = argparse.ArgumentParser("Running GemNet on ppi mutation change prediction task.")
    parser.add_argument("--config", type=str, default="./configs/s4169.yaml", 
                        help="which config file to use")
    parser.add_argument("--local_rank", type=int)
    args, other_args = parser.parse_known_args()
    return args, other_args

    # Used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))

def build_dataset(config):
    class_name = config["dataset_class"]
    return getattr(data_containers, class_name).from_config(config)

if __name__ == "__main__":
    args, other_args = parse_args()

    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("INFO")

    # TODO: update config with other args.

    with open(args.config, 'r') as c:
        config = yaml.safe_load(c)

    # For strings that yaml doesn't parse (e.g. None)
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
    
    config = EasyDict(config)
    config.model["num_targets"] = 2 if config.model.mve else 1
    torch.manual_seed(config.tfseed)

    logging.info("Start training")
    num_gpus = torch.cuda.device_count()
    cuda_available = torch.cuda.is_available()
    logging.info(f"Available GPUs: {num_gpus}")
    logging.info(f"CUDA Available: {cuda_available}")
    if num_gpus == 0:
        logging.warning("No GPUs were found. Training is run on CPU!")
    if not cuda_available:
        logging.warning("CUDA unavailable. Training is run on CPU!")

    if (config.restart is None) or (config.restart == "None"): 
        directory = config.logdir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id_generator() + "_" + "_" + os.path.basename(args.config)
    else:
        directory = config.restart
    
    logging.info(f"Directory: {directory}")
    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    best_dir = os.path.join(directory, "best")
    if not os.path.exists(best_dir):
        os.makedirs(best_dir, exist_ok=True)
    log_dir = os.path.join(directory, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(directory, "log.txt"))
    logger.addHandler(fh)    

    extension = ".pth"
    log_path_model = f"{log_dir}/model{extension}"
    log_path_training = f"{log_dir}/training{extension}"
    best_path_model = f"{best_dir}/model{extension}"

    logging.info("Initializing model")
    model = GemNet(**config.model)

    logging.info("Initializing trainer")

    # Initialize trainer
    trainer_class = config.trainer.pop("class")
    trainer = getattr(trainers, trainer_class)(model=model, **config.trainer)

    # Initialize metrics
    train_metrics = Metrics("train", trainer.tracked_metrics)
    val_metrics = Metrics("val", trainer.tracked_metrics)
    test_metrics = Metrics("test", trainer.tracked_metrics)

    # Save/load best recorded loss (only the best model is saved)
    metrics_best_val = BestMetrics(best_dir, val_metrics, main_metric=config.main_metric, 
                            metric_mode=config.metric_mode)
    metrics_best_test = BestMetrics(best_dir, test_metrics, main_metric=config.main_metric,
                            metric_mode=config.metric_mode)

    # Set up checkpointing
    # Restore latest checkpoint
    if os.path.exists(log_path_model):
        logging.info("Restoring model and trainer")
        model_checkpoint = torch.load(log_path_model)
        model.load_state_dict(model_checkpoint["model"])

        train_checkpoint = torch.load(log_path_training)
        trainer.load_state_dict(train_checkpoint["trainer"])
        # restore the best saved results
        metrics_best_val.restore()
        logging.info(f"Restored best metrics: {metrics_best_val.loss}")
        step_init = int(train_checkpoint["step"])
    else:
        logging.info("Freshly initialize model")
        metrics_best_val.inititalize()
        step_init = 0

    for epoch in tqdm(range(config.num_epochs)):
        # prepare data, support re
        all_data_path = list()
        for path_str in config.dataset:
            dir_name = os.path.dirname(path_str)
            file_name = os.path.basename(path_str)
            dir_path_obj = Path(dir_name)
            matched_files = list(dir_path_obj.rglob(file_name))
            matched_files = [str(x) for x in matched_files if os.path.exists(str(x))]
            all_data_path.extend(matched_files)
        all_data_path = list(set(all_data_path))

        for data_path in all_data_path:
            ## process re

            new_config = copy.deepcopy(config)
            new_config["dataset"] = data_path

            data_container = build_dataset(new_config)

            logger.info(f"data path {new_config.dataset} dataset length: {len(data_container)}")

            data_provider = DataProvider(data_container, len(data_container), 0,
                                        config.batch_size, seed=config.data_seed, shuffle=True, 
                                        random_split=False)
            # Perform training step
            trainer.train_on_epoch(data_provider, train_metrics, config.iter_per_epoch)
        # Save progress
        if epoch % config.save_interval == 0:
            torch.save({"model": model.state_dict()}, log_path_model)
            torch.save(
                {"trainer": trainer.state_dict(), "step": epoch}, log_path_training
            )


        # Save backup variables and load averaged variables
        trainer.save_variable_backups()
        trainer.load_averaged_variables()

        train_metrics_res = train_metrics.result(append_tag=False)
        metrics_strings = [f"{key}: train={train_metrics_res[key]:.6f}" for key in train_metrics.keys]
        if trainer.rank==0:
            logging.info(
                f"epoch ({epoch}): " + "; ".join(metrics_strings)
            )

        # decay learning rate on plateau
        trainer.decay_maybe(train_metrics.loss)

        train_metrics.reset_states()

        # Restore backup variables
        trainer.restore_variable_backups()
