import argparse
import os, sys
from gemnet.training import runners
import yaml
import ast
from tqdm import tqdm
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

def parse_args():
    parser = argparse.ArgumentParser("Running GemNet on ppi mutation change prediction task.")
    parser.add_argument("--config", "-c", type=str, default="./configs/s4169.yaml", 
                        help="which config file to use")
    parser.add_argument("--save_path", type=str, default="./save_feature/")
    parser.add_argument("--local_rank", type=int, default=0)
    args, other_args = parser.parse_known_args()
    return args, other_args


def feature_on_batch(runner, batch):
    inputs, targets = batch
    wild_type, mutant = inputs
    targets = targets[0]
    wild_type, mutant, targets = runner.trainer.dict2device(wild_type), runner.trainer.dict2device(mutant), runner.trainer.dict2device(targets)

    energy, wild_output_dict, mutant_output_dict = runner.model(wild_type, mutant, return_output=True)
    return wild_output_dict["atom_feature"], mutant_output_dict["atom_feature"]


def save_feature(runner, split="test", save_path="./save_feature"):
    """
    1. atom identifer: mutant | context 
    """
    provider = runner.data_provider
    model = runner.model
    model.eval()
    
    wild_type_atom_features, mutant_atom_features = list(), list()
    wild_type_is_mutant_residue, mutant_is_mutant_residue = list(), list()
    loader = provider.get_loader(split)
    
    with torch.no_grad():
        for batch in tqdm(loader):
            wt_feature, mt_feature = feature_on_batch(runner, batch) 
            # wt_feature: list of tensor of size [N, D]
            wild_type_atom_features.append(wt_feature)
            mutant_atom_features.append(mt_feature)
            ## extract "is mutant residue" label
            wild_type, mutant = batch[0]
            wild_type_is_mutant_residue.append(wild_type["is_mutant_residue"])
            mutant_is_mutant_residue.append(mutant["is_mutant_residue"])

    num_outputs = len(wild_type_atom_features[0])

    wild_type_atom_features = [torch.cat([x[i] for x in wild_type_atom_features], dim=0).cpu() for i in range(num_outputs)]
    mutant_atom_features = [torch.cat([x[i] for x in mutant_atom_features], dim=0).cpu() for i in range(num_outputs)]
    
    wild_type_is_mutant_residue = torch.stack(wild_type_is_mutant_residue, dim=0).cpu()
    mutant_is_mutant_residue = torch.stack(mutant_is_mutant_residue, dim=0).cpu()

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    torch.save(dict(wt=wild_type_atom_features, mt=mutant_atom_features, 
                    wt_label=wild_type_is_mutant_residue, mt_label=mutant_is_mutant_residue), os.path.join(save_path, "features.pth"))
    

if __name__ == "__main__":
    ## load runner
    args, other_args = parse_args()
    with open(args.config, 'r') as c:
        config = yaml.safe_load(c)
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
    runner_class = config.get("runner", "DownStreamRunner")
    runner = getattr(runners, runner_class)(args, other_args)

    save_feature(runner, "test", args.save_path)
    