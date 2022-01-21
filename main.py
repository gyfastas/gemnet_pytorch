import argparse
import os, sys
from gemnet.training import runners
import yaml
import ast
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"


def parse_args():
    parser = argparse.ArgumentParser("Running GemNet on ppi mutation change prediction task.")
    parser.add_argument("--config", "-c", type=str, default="./configs/s4169.yaml", 
                        help="which config file to use")
    parser.add_argument("--local_rank", type=int)
    args, other_args = parser.parse_known_args()
    return args, other_args

if __name__ == "__main__":
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
    runner.run()