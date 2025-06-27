import argparse
from easydict import EasyDict
import numpy as np
import random
import torch
import yaml

from mid import MID
from joint_pred_mid import JointPredMID
from constant_velocity_baseline import ConstantVelocityBaseline
from standing_baseline import StandingBaseline

import matplotlib

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of MID")
    parser.add_argument("--config", default="")
    parser.add_argument("--test_dataset", default=None)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="visualize velocity field (velocity field of agent velocities)",
    )
    return parser.parse_args()


def set_random_seed(seed):
    if seed < 0:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config = EasyDict(config)
    return config


def run_mid(config, args, seed):
    agent = MID(config, test_dataset=args.test_dataset, time=config["time"])
    if True or config["eval_mode"]:
        agent.eval(config["sampling"], config["num_steps"])
    else:
        agent.train(config["sampling"], config["num_steps"], seed)


def run_joint_pred_mid(config, args, seed):
    agent = JointPredMID(config, test_dataset=args.test_dataset, time=config["time"])
    if config["eval_mode"]:
        agent.eval(config["sampling"], config["num_steps"], plot=True)
    else:
        agent.train(config["sampling"], config["num_steps"], seed)


def run_standing_baseline(config, args):
    agent = StandingBaseline(config, test_dataset=args.test_dataset)
    agent.eval()


def run_constant_velocity_baseline(config, args):
    agent = ConstantVelocityBaseline(config, test_dataset=args.test_dataset)
    agent.eval(None, None)


def main():
    args = parse_args()
    config = load_config(args)
    seed = set_random_seed(config["seed"])
    if config["method"] == "mid":
        run_mid(config, args, seed)
    elif config["method"] == "mid_jp":
        run_joint_pred_mid(config, args, seed)
    elif config["method"] == "standing_baseline":
        run_standing_baseline(config, args)
    elif config["method"] == "constant_velocity_baseline":
        run_constant_velocity_baseline(config, args)


if __name__ == "__main__":
    # import debugpy

    # # Start the debugpy server
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()

    main()
