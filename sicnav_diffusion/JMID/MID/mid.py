import dill
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pickle
import scipy
from easydict import EasyDict
import yaml
import sys

from tensorboardX import SummaryWriter
import time
import torch
from torch import optim, utils
from tqdm.auto import tqdm

# os.environ["WANDB_MODE"] = "disabled"
# import wandb
try:
    from dataset import (
        EnvironmentDataset,
        collate,
        joint_pred_train_collate,
        get_timesteps_data,
        get_timesteps_each_scene_data,
        get_scenes_in_att_rad,
    )
    from environment import environment
    from evaluation import evaluation
    from models.autoencoder import AutoEncoder
    import models.diffusion as diffusion
    from models.diffusion import DiffusionTraj, VarianceSchedule
    from models.encoders.model_utils import ModeKeys
    from models.trajectron import Trajectron
    from utils.model_registrar import ModelRegistrar
    from utils.trajectron_hypers import get_traj_hypers
    from dataset.preprocessing import generate_mask, masked_fill, get_list_data_items_from_batch
except ImportError:
    from .dataset import (
        EnvironmentDataset,
        collate,
        joint_pred_train_collate,
        get_timesteps_data,
        get_timesteps_each_scene_data,
        get_scenes_in_att_rad,
    )
    from .environment import environment
    from .evaluation import evaluation
    from .models.autoencoder import AutoEncoder
    from .models import diffusion as diffusion
    from .models.diffusion import DiffusionTraj, VarianceSchedule
    from .models.encoders.model_utils import ModeKeys
    from .models.trajectron import Trajectron
    from .utils.model_registrar import ModelRegistrar
    from .utils.trajectron_hypers import get_traj_hypers
    from .dataset.preprocessing import generate_mask, masked_fill, get_list_data_items_from_batch


import copy
import shutil

EARLY_STOP_PATIENCE = 10  # epochs
SAVE_ITERS_FOR_TRAIN_LOSS = 2000 # 100 # 2000  # iters

IS_SCENE_IN_ATTENTION_RADIUS = True
IS_DATALOADER_DEBUG = False # True
DEBUG = False


class MID:
    def __init__(
        self,
        config,
        test_dataset=None,
        is_joint_pred=False,
        time=False,
        sicnav_inference=False,
        init_env=None,
        num_history=None,
        prediction_horizon=None,
    ):
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)
            self.config = EasyDict(config)
        else:
            self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_dataset = test_dataset
        self.is_joint_pred = is_joint_pred
        self.time = time
        self.sicnav_inference = sicnav_inference

        torch.backends.cudnn.benchmark = True
        self._build(init_env=init_env, num_history=num_history, prediction_horizon=prediction_horizon)
        if self.sicnav_inference:
            self.eval = self.eval_sicnav
            self.num_samples = self.config["num_samples"]
            self.time = self.config["time"]
        else:
            self.eval = self.eval_not_sicnav

    def train(self, sampling, step, seed):
        # wandb.login()
        config_name = self.config.config[self.config.config.find("/")+1:self.config.config.rfind(".yaml")]
        # run = wandb.init(
        #     name=config_name,
        #     project="MID_Training",
        #     entity="sepsam",
        #     config=self.config,
        # )

        best_val_ade = float("inf")
        patience = 0  # for early stopping
        total_count_iter = 0
        total_grad_steps = -1
        train_losses_for_iter_vis = []
        for epoch in range(1, self.config.epochs + 1):
            train_losses = []
            self.model.train()
            self.train_dataset.augment = self.config.augment
            total_agent_num = 0
            count_iter = 0
            max_agent_num = 0
            min_agent_num = float("inf")
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch_w_batch_size in pbar:
                    total_grad_steps += 1
                    if self.is_joint_pred:
                        batch = batch_w_batch_size[0]
                        batch_size = batch_w_batch_size[1]

                        total_agent_num += int(batch[0].shape[0] / batch_size)
                        count_iter += 1
                        total_count_iter += 1
                        if int(batch[0].shape[0] / batch_size) > max_agent_num:
                            max_agent_num = int(batch[0].shape[0] / batch_size)
                        if int(batch[0].shape[0] / batch_size) < min_agent_num:
                            min_agent_num = int(batch[0].shape[0] / batch_size)
                    else:
                        batch = batch_w_batch_size

                    self.optimizer.zero_grad()

                    if self.is_joint_pred:
                        attn_mask, loss_mask = generate_mask(batch, batch_size, self.hyperparams["prediction_horizon"])
                        if IS_DATALOADER_DEBUG:
                            list_data_items = get_list_data_items_from_batch(batch, batch_size)
                            for batch_id, data_item in enumerate(list_data_items):
                                self._plot_predictions(data_item, save_name="trans_bid{}_data.pdf".format(batch_id))
                                self._plot_predictions(
                                    data_item, is_vis_st=True, save_name="trans_st_bid{}_data.pdf".format(batch_id)
                                )
                            self._dump_mask(attn_mask, loss_mask)
                        # Nan must be converted to 0.
                        # https://discuss.pytorch.org/t/nn-transformerencoder-all-nan-values-when-src-key-padding-mask-provided/131157/5
                        batch = masked_fill(batch)

                        train_loss = self.model.get_loss(ModeKeys.TRAIN, batch, node_type,
                                                        batch_size=batch_size,
                                                        attn_mask=attn_mask.cuda(),
                                                        loss_mask=loss_mask.cuda())
                    else:
                        train_loss = self.model.get_loss(ModeKeys.TRAIN, batch, node_type)
                    del batch
                    torch.cuda.empty_cache()
                    train_losses.append(train_loss.item())
                    train_losses_for_iter_vis.append(train_loss.item())
                    pbar.set_description(
                        f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}"
                    )

                    # wandb.log({
                    #             "Train Loss (MSE)": train_loss.item(),
                    #           }, step=total_grad_steps)
                    train_losses_for_iter_vis = []
                    train_loss.backward()
                    self.optimizer.step()


            if count_iter > 0:
                print(
                    "Average Agent Number:{}, Max Agent Number:{}, Min Agent Number:{}".format(
                        int(total_agent_num / count_iter), max_agent_num, min_agent_num
                    )
                )
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = self.validation()
            log = {"Avg. Train Loss per Epoch (MSE)": avg_train_loss, "Avg. Val Loss per Epoch (MSE)": avg_val_loss, "Epoch": epoch}
            # log = {"Avg. Train Loss per Epoch (MSE)": avg_train_loss, "Epoch": epoch}
            # wandb.log(log, step=total_grad_steps)

            self.train_dataset.augment = False  # TODO: Why is this needed?

            self._save_model(epoch)

            if epoch % self.config.eval_every == 0:
                ade, fde, kde, ade_most_likely, fde_most_likely, sade, sfde, sade_mean, sfde_mean = self.eval(
                    sampling, step, epoch=epoch
                )
                if not self.config.fast_train:
                    log = {
                        # "Train Loss (MSE)": avg_train_loss,
                        "Validation min of 20 ADE (m)": ade,
                        "Validation min of 20 FDE (m)": fde,
                        "Validation KDE NLL": kde,
                        "Validation most likely ADE (m)": ade_most_likely,
                        "Validation most likely FDE (m)": fde_most_likely,
                        "Validation min of 20 SADE (m)": sade,
                        "Validation min of 20 SFDE (m)": sfde,
                        "Validation mean SADE (m)": sade_mean,
                        "Validation mean SFDE (m)": sfde_mean,
                        "Validation Steps": epoch,
                    }
                    best_val_ade, was_model_saved = self._save_model_if_best(
                        ade, best_val_ade, epoch
                    )
                else:
                    log = {
                        # "Train Loss (MSE)": avg_train_loss,
                        "Validation min of 20 ADE (m)": ade,
                        "Validation min of 20 FDE (m)": fde,
                        "Validation KDE NLL": kde,
                        "Validation mean ADE (m)": ade_most_likely,
                        "Validation mean FDE (m)": fde_most_likely,
                        "Validation min of 20 SADE (m)": sade,
                        "Validation min of 20 SFDE (m)": sfde,
                        "Validation mean SADE (m)": sade_mean,
                        "Validation mean SFDE (m)": sfde_mean,
                        "Validation Steps": epoch,
                    }
                    best_val_ade, was_model_saved = self._save_model_if_best(
                        ade_most_likely, best_val_ade, epoch
                    )
                # wandb.log(log, step=total_grad_steps)
                self.model.train()

                if not self.config.early_stopping:
                    continue
                if was_model_saved:  # Only works if self.config.eval_every is 1
                    patience = 0
                else:
                    patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    # Stop training
                    break

    def validation(self):
        self.model.eval()
        losses = []
        total_agent_num = 0
        count_iter = 0
        max_agent_num = 0
        min_agent_num = float("inf")
        with torch.no_grad():
            for node_type, data_loader in self.eval_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch_w_batch_size in pbar:
                    if self.is_joint_pred:
                        batch = batch_w_batch_size[0]
                        batch_size = batch_w_batch_size[1]
                        total_agent_num += int(batch[0].shape[0] / batch_size)
                        count_iter += 1
                        if int(batch[0].shape[0] / batch_size) > max_agent_num:
                            max_agent_num = int(batch[0].shape[0] / batch_size)
                        if int(batch[0].shape[0] / batch_size) < min_agent_num:
                            min_agent_num = int(batch[0].shape[0] / batch_size)
                    else:
                        batch = batch_w_batch_size

                    if self.is_joint_pred:
                        attn_mask, loss_mask = generate_mask(batch, batch_size, self.hyperparams["prediction_horizon"])
                        batch = masked_fill(batch)
                        loss = self.model.get_loss(ModeKeys.EVAL, batch, node_type,
                                                        batch_size=batch_size,
                                                        attn_mask=attn_mask.cuda(),
                                                        loss_mask=loss_mask.cuda())
                    else:
                        loss = self.model.get_loss(ModeKeys.EVAL, batch, node_type)
                    del batch
                    torch.cuda.empty_cache()
                    losses.append(loss.item())
                    pbar.set_description(f"Validation MSE: {loss.item():.2f}")
        avg_loss = sum(losses) / len(losses)
        if count_iter > 0:
            print(
                "Average Agent Number:{}, Max Agent Number:{}, Min Agent Number:{}".format(
                    int(total_agent_num / count_iter), max_agent_num, min_agent_num
                )
            )

        return avg_loss

    def eval_sicnav(self, eval_env):
        self._build_eval_loader_sicnav_inference(eval_env)
        self.model.eval()

        node_type = "PEDESTRIAN"

        inference_scene = self.eval_scenes[0]  # There should only be one scene
        inference_env = self.eval_env

        timestep_step_size = 1
        assert timestep_step_size == 1, "For joint prediction, timestep_step_size should be 1."
        t = self.hyperparams["maximum_history_length"]  # set to num_history timestep
        timesteps = np.arange(t, t + timestep_step_size)
        # timesteps = torch.arange(t, t + timestep_step_size, device=self.device)
        batch = get_timesteps_data(
            env=inference_env,
            scene=inference_scene,
            t=timesteps,
            node_type=node_type,
            state=self.hyperparams["state"],
            pred_state=self.hyperparams["pred_state"],
            edge_types=inference_env.get_edge_types(),
            min_ht=self.hyperparams["maximum_history_length"],
            max_ht=self.hyperparams["maximum_history_length"],
            min_ft=0,
            max_ft=0,
            hyperparams=self.hyperparams,
        )
        test_batch = batch[0]
        nodes = batch[1]
        timesteps_o = batch[2]
        traj_pred, num_steps_for_all_samples = self.model.generate_sicnav_inference(
            test_batch,
            node_type,
            num_points=self.hyperparams["prediction_horizon"],
            sample=self.num_samples,
            bestof=True,
            sampling="ddim",
            step=self.config.step_size,
            with_constraints=False,
        )  # 20 * B * 12 * 2

        preds = self._get_preds_sicnav_inference(
                traj_pred,
                nodes,
                timesteps_o,
                inference_scene.normalized_px,
                inference_scene.img_width,
                inference_scene.img_height,
            )

        return preds


    def eval_not_sicnav(self, sampling, step, with_constraints=False, epoch=None, plot=False):
        """`epoch` is just for printing purposes"""
        print_num_samples_with_collisions = False
        self.model.eval()

        if epoch is None:
            epoch = self.config.eval_at
        if sampling == "ddim":
            self.log.info(f"Sampling: {sampling} Num of steps: {step}")

        node_type = "PEDESTRIAN"

        eval_ade_batch_errors = []
        eval_ade_most_likely_batch_errors = []
        eval_mean_ade_batch_errors = []
        eval_std_ade_batch_errors = []
        eval_fde_batch_errors = []
        eval_fde_most_likely_batch_errors = []
        eval_mean_fde_batch_errors = []
        eval_std_fde_batch_errors = []
        eval_kde_batch_errors = []
        eval_ade_list_batch_errors = []
        eval_fde_list_batch_errors = []
        eval_de_list_batch_errors = []
        eval_sade_batch_errors = []
        eval_mean_sade_batch_errors = []
        eval_std_sade_batch_errors = []
        eval_sfde_batch_errors = []
        eval_mean_sfde_batch_errors = []
        eval_std_sfde_batch_errors = []
        if self.config.is_eval_hst:
            eval_ade_one_fourth_batch_errors = []
            eval_ade_two_fourth_batch_errors = []
            eval_ade_three_fourth_batch_errors = []
            eval_ade_most_likely_one_fourth_batch_errors = []
            eval_ade_most_likely_two_fourth_batch_errors = []
            eval_ade_most_likely_three_fourth_batch_errors = []
            eval_ade_mean_one_fourth_batch_errors = []
            eval_ade_mean_two_fourth_batch_errors = []
            eval_ade_mean_three_fourth_batch_errors = []
            eval_kde_one_fourth_batch_errors = []
            eval_kde_two_fourth_batch_errors = []
            eval_kde_three_fourth_batch_errors = []

        if IS_DATALOADER_DEBUG:
            data_items = []

        ph = self.hyperparams["prediction_horizon"]
        max_hl = self.hyperparams["maximum_history_length"]
        total_num_gt_tracks = 0
        total_num_det_tracks = 0
        all_inference_times = []
        all_num_iters = []
        num_steps_for_all_samples_list = []
        total_num_samples_with_collisions = 0
        total_num_samples_with_collisions_fixed = 0
        for i, scene in enumerate(self.eval_scenes):
            print(
                f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}: {scene.name} -----"
            )
            inference_times = []
            num_iters = []

            total_num_gt_tracks += len(scene.nodes)
            inference_scene = scene  # Initialize
            inference_env = self.eval_env
            edge_types = [
                edge_type
                for edge_type in inference_env.get_edge_types()
                if edge_type[0].name == node_type
            ]

            inference_node = scene.nodes[0]
            inference_node_id_to_gt_node = {node.id: node for node in scene.nodes}
            inference_scene = self._get_inference_scene(scene)
            inference_env = self.inference_env
            inference_nodes = inference_scene.nodes
            inference_node = inference_nodes[0]
            gt_nodes = scene.nodes
            inference_idx, gt_idx = self._associate_nodes_with_other_nodes(
                inference_nodes,
                gt_nodes,
                inference_scene.normalized_px,  # Assume that inference_scene and scene both are or are not using normalized pixels
                inference_scene.img_width,  # Assume width and height of images are the same between inference_scene and scene if they are images
                inference_scene.img_height,
            )
            inference_node_id_to_gt_node = {
                inference_nodes[inference_idx[i]].id: gt_nodes[gt_idx[i]]
                for i in range(len(inference_idx))
            }
            total_num_det_tracks += len(inference_node_id_to_gt_node)

            if self.config.save_trajectories:
                time_to_node_id_to_preds = {}
                if self.config.is_eval_hst:
                    time_to_node_id_to_interpolated_infos = {}
                    time_to_node_id_to_interpolated_infos["future"] = {}
                    time_to_node_id_to_interpolated_infos["history"] = {}
            timestep_step_size = self.config.eval_timesteps
            inf_timesteps = range(0, inference_scene.timesteps, timestep_step_size)
            if self.time or with_constraints:
                timestep_step_size = 1
                inf_timesteps = range(0, inference_scene.timesteps)
            for t in tqdm(inf_timesteps):
                timesteps = np.arange(t, t + timestep_step_size)
                if not IS_SCENE_IN_ATTENTION_RADIUS:
                    batch = get_timesteps_data(
                        env=inference_env,
                        scene=inference_scene,
                        t=timesteps,
                        node_type=node_type,
                        state=self.hyperparams["state"],
                        pred_state=self.hyperparams["pred_state"],
                        edge_types=edge_types,
                        min_ht=max_hl,
                        max_ht=max_hl,
                        min_ft=ph,
                        max_ft=ph,
                        hyperparams=self.hyperparams,
                    )
                    if batch is None:
                        continue
                    test_batch = batch[0]
                    nodes = batch[1]
                    timesteps_o = batch[2]
                    if self.time:
                        start = time.time()
                    traj_pred, num_steps_for_all_samples = self.model.generate(
                        test_batch,
                        node_type,
                        num_points=ph,
                        sample=self.config.num_samples,
                        bestof=True,
                        sampling=sampling,
                        step=step,
                        with_constraints=self.config.with_constraints,
                        constraint_type=self.config.constraint_type,
                    )  # 20 * B * 12 * 2
                    if self.time:
                        inference_times.append(time.time() - start)
                        num_iters.append(num_iter)
                        continue
                    num_steps_for_all_samples_list.append(num_steps_for_all_samples)
                    total_num_samples_with_collisions += num_samples_with_collisions
                    total_num_samples_with_collisions_fixed += (
                        num_samples_with_collisions_fixed
                    )
                    if IS_DATALOADER_DEBUG:
                        self._plot_predictions(
                            test_batch,
                            names=nodes,
                            predictions=traj_pred,
                            save_name="eval.jpg",
                        )
                else:
                    start = time.time()
                    scenes, are_scenes_unique = get_scenes_in_att_rad(
                        env=inference_env,
                        scene=inference_scene,
                        t=timesteps,
                        node_type=node_type,
                        state=self.hyperparams["state"],
                        pred_state=self.hyperparams["pred_state"],
                        edge_types=edge_types,
                        min_ht=max_hl,
                        max_ht=max_hl,
                        min_ft=ph,
                        max_ft=ph,
                        hyperparams=self.hyperparams,
                        gen_one_scene_per_node=True,
                    )
                    if scenes is None:
                        continue
                    if IS_DATALOADER_DEBUG:
                        all_test_batch = []
                    for s, scene in enumerate(scenes):
                        # First node in scene.nodes is the target node
                        assert scene.nodes[0].type.name == node_type, "The target agent is not a pedestrian"
                        batch = get_timesteps_each_scene_data(
                            env=inference_env,
                            scene=scene,
                            t=timesteps,
                            node_type=node_type,
                            state=self.hyperparams["state"],
                            pred_state=self.hyperparams["pred_state"],
                            edge_types=edge_types,
                            min_ht=max_hl,
                            max_ht=max_hl,
                            min_ft=ph,
                            max_ft=ph,
                            hyperparams=self.hyperparams,
                        )
                        if batch is None:
                            continue
                        local_test_batch = batch[0]
                        local_nodes = batch[1]
                        local_timesteps_o = batch[2]
                        if self.time:
                            start = time.time()
                        if DEBUG:
                            if len(local_nodes) > 1:
                                seed = 0
                                import random
                                random.seed(seed)
                                np.random.seed(seed)
                                torch.manual_seed(seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed(seed)
                                    torch.cuda.manual_seed_all(seed)
                        (
                            local_traj_pred_torch,
                            num_steps_for_all_samples,
                            num_iter,
                            num_samples_with_collisions,
                            num_samples_with_collisions_fixed,
                        ) = self.model.generate(
                            local_test_batch,
                            node_type,
                            num_points=ph,
                            sample=self.config.num_samples,
                            bestof=True,
                            sampling=sampling,
                            step=step,
                            with_constraints=self.config.with_constraints,
                            constraint_type=self.config.constraint_type,
                            as_numpy=False,
                        )  # 20 * B * 12 * 2
                        if self.time:
                            inference_times.append(time.time() - start)
                            continue
                        local_traj_pred = local_traj_pred_torch.cpu().detach().numpy()
                        num_steps_for_all_samples_list.append(num_steps_for_all_samples)
                        if DEBUG:
                            # Check to make sure results of joint predictions with more than one node are identical, regardless of the target agent
                            if len(local_nodes) > 1:
                                if not os.path.isdir("./tmp"):
                                    os.makedirs("./tmp")
                                if not os.path.isdir("./tmp_norm"):
                                    os.makedirs("./tmp_norm")
                                self._plot_predictions(
                                    local_test_batch,
                                    names=local_nodes,
                                    predictions_torch=local_traj_pred_torch,
                                    save_name="./tmp/eval_"
                                    + str(scene.name)
                                    + "_"
                                    + str(t).zfill(3)
                                    + "_"
                                    + str(s).zfill(3)
                                    + "_local.jpg",
                                )
                                self._plot_predictions(
                                    local_test_batch,
                                    is_vis_st=True,
                                    names=local_nodes,
                                    predictions_torch=local_traj_pred_torch,
                                    save_name="./tmp_norm/eval_st_"
                                    + str(scene.name)
                                    + "_"
                                    + str(t).zfill(3)
                                    + "_"
                                    + str(s).zfill(3)
                                    + "_local.jpg",
                                )
                        traj_pred = local_traj_pred
                        nodes = local_nodes
                        timesteps_o = local_timesteps_o
                        predictions = traj_pred
                        predictions_dict = {}
                        for i, ts in enumerate(
                            timesteps_o
                        ):  # loop over the agents in the scene
                            if ts not in predictions_dict.keys():
                                predictions_dict[ts] = dict()
                            gt_node = inference_node_id_to_gt_node.get(nodes[i].id)

                            # Skip extra detected tracks (detected tracks that don't overlap with any ground truth tracks in time)
                            # if not isinstance(gt_node, environment.Node):
                            #     continue

                            # Only evaluate ADE when the ground truth exists now (at ts) to be consistent with OSPA(2) calculations
                            if not self.exists_at_t(gt_node, ts):
                                continue

                            predictions_dict[ts][gt_node] = np.transpose(
                                predictions[:, [i]], (1, 0, 2, 3)  # B x num_samples x ph x 2
                            )

                        gt_node = inference_node_id_to_gt_node[
                            list(inference_node_id_to_gt_node.keys())[0]
                        ]  # Use any gt node since we assume they all have the same data preprocessing
                        (
                            batch_error_dict,
                            interpolated_futures_dict,
                            interpolated_histories_dict,
                        ) = evaluation.compute_batch_statistics(
                            predictions_dict,
                            scene.dt,  # Assume scene.dt is equal to inference_scene.dt
                            max_hl,
                            ph,
                            self.eval_env.NodeType,  # Assume self.eval_env.NodeType is equal to self.inference_env.NodeType
                            gt_node.get_mean_x_and_y(),
                            inference_node.get_mean_x_and_y(),  # Use any inference node since we assume they all have the same mean x and mean y
                            target_node_id=nodes[0].id,
                            kde=not self.config.fast_train,
                            compute_ml=not self.config.fast_train,
                            map=None,
                            best_of=True,
                            all_de=True,
                            prune_ph_to_future=True,
                            normalized_px=inference_scene.normalized_px,
                            img_width=inference_scene.img_width,
                            img_height=inference_scene.img_height,
                            is_eval_hst=self.config.is_eval_hst,
                        )
                        if DEBUG:
                            if len(local_nodes) > 1:
                                print("Timestep:", timesteps[0])
                                print("Target agent:", nodes[0].id)
                                print("Agents:", predictions_dict[ts].keys())
                                print("SADE min:", batch_error_dict[node_type]["sade"])
                                print("SADE mean:", batch_error_dict[node_type]["sade_mean"])
                                print("SADE std:", batch_error_dict[node_type]["sade_std"])
                                print("SFDE min:", batch_error_dict[node_type]["sfde"])
                                print("SFDE mean:", batch_error_dict[node_type]["sfde_mean"])
                                print("SFDE std:", batch_error_dict[node_type]["sfde_std"])
                        eval_ade_batch_errors = np.hstack(
                            (eval_ade_batch_errors, batch_error_dict[node_type]["ade"])
                        )
                        eval_ade_most_likely_batch_errors = np.hstack(
                            (
                                eval_ade_most_likely_batch_errors,
                                batch_error_dict[node_type]["ade_most_likely"],
                            )
                        )
                        eval_mean_ade_batch_errors = np.hstack(
                            (
                                eval_mean_ade_batch_errors,
                                batch_error_dict[node_type]["ade_mean"],
                            )
                        )
                        eval_std_ade_batch_errors = np.hstack(
                            (eval_std_ade_batch_errors, batch_error_dict[node_type]["ade_std"])
                        )
                        eval_fde_batch_errors = np.hstack(
                            (eval_fde_batch_errors, batch_error_dict[node_type]["fde"])
                        )
                        eval_fde_most_likely_batch_errors = np.hstack(
                            (
                                eval_fde_most_likely_batch_errors,
                                batch_error_dict[node_type]["fde_most_likely"],
                            )
                        )
                        eval_mean_fde_batch_errors = np.hstack(
                            (
                                eval_mean_fde_batch_errors,
                                batch_error_dict[node_type]["fde_mean"],
                            )
                        )
                        eval_std_fde_batch_errors = np.hstack(
                            (eval_std_fde_batch_errors, batch_error_dict[node_type]["fde_std"])
                        )
                        eval_kde_batch_errors = np.hstack(
                            (eval_kde_batch_errors, batch_error_dict[node_type]["kde"])
                        )
                        eval_ade_list_batch_errors = np.hstack(
                            (
                                eval_ade_list_batch_errors,
                                batch_error_dict[node_type]["ade_list"],
                            )
                        )
                        eval_fde_list_batch_errors = np.hstack(
                            (
                                eval_fde_list_batch_errors,
                                batch_error_dict[node_type]["fde_list"],
                            )
                        )
                        eval_de_list_batch_errors = np.hstack(
                            (eval_de_list_batch_errors, batch_error_dict[node_type]["de_list"])
                        )
                        if self.config.is_eval_hst:
                            eval_ade_one_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_one_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_one_fourth"],
                                )
                            )
                            eval_ade_two_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_two_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_two_fourth"],
                                )
                            )
                            eval_ade_three_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_three_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_three_fourth"],
                                )
                            )
                            eval_ade_most_likely_one_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_most_likely_one_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_most_likely_one_fourth"],
                                )
                            )
                            eval_ade_most_likely_two_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_most_likely_two_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_most_likely_two_fourth"],
                                )
                            )
                            eval_ade_most_likely_three_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_most_likely_three_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_most_likely_three_fourth"],
                                )
                            )
                            eval_ade_mean_one_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_mean_one_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_mean_one_fourth"],
                                )
                            )
                            eval_ade_mean_two_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_mean_two_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_mean_two_fourth"],
                                )
                            )
                            eval_ade_mean_three_fourth_batch_errors = np.hstack(
                                (
                                    eval_ade_mean_three_fourth_batch_errors,
                                    batch_error_dict[node_type]["ade_mean_three_fourth"],
                                )
                            )
                            eval_kde_one_fourth_batch_errors = np.hstack(
                                (
                                    eval_kde_one_fourth_batch_errors,
                                    batch_error_dict[node_type]["kde_one_fourth"],
                                )
                            )
                            eval_kde_two_fourth_batch_errors = np.hstack(
                                (
                                    eval_kde_two_fourth_batch_errors,
                                    batch_error_dict[node_type]["kde_two_fourth"],
                                )
                            )
                            eval_kde_three_fourth_batch_errors = np.hstack(
                                (
                                    eval_kde_three_fourth_batch_errors,
                                    batch_error_dict[node_type]["kde_three_fourth"],
                                )
                            )
                            time_to_node_id_to_interpolated_infos["future"].update(
                                interpolated_futures_dict
                            )
                            time_to_node_id_to_interpolated_infos["history"].update(
                                interpolated_histories_dict
                            )
                        is_scene_unique = are_scenes_unique[s]
                        # is_scene_unique = True
                        if is_scene_unique:
                            eval_sade_batch_errors = np.hstack(
                                (eval_sade_batch_errors, batch_error_dict[node_type]["sade"])
                            )
                            eval_mean_sade_batch_errors = np.hstack(
                                (
                                    eval_mean_sade_batch_errors,
                                    batch_error_dict[node_type]["sade_mean"],
                                )
                            )
                            eval_std_sade_batch_errors = np.hstack(
                                (eval_std_sade_batch_errors, batch_error_dict[node_type]["sade_std"])
                            )
                            eval_sfde_batch_errors = np.hstack(
                                (eval_sfde_batch_errors, batch_error_dict[node_type]["sfde"])
                            )
                            eval_mean_sfde_batch_errors = np.hstack(
                                (
                                    eval_mean_sfde_batch_errors,
                                    batch_error_dict[node_type]["sfde_mean"],
                                )
                            )
                            eval_std_sfde_batch_errors = np.hstack(
                                (eval_std_sfde_batch_errors, batch_error_dict[node_type]["sfde_std"])
                            )
                        if IS_DATALOADER_DEBUG:
                            if not os.path.isdir("./tmp_"+scene.name):
                                os.makedirs("./tmp_"+scene.name)
                            if not os.path.isdir("./tmp_norm_"+scene.name):
                                os.makedirs("./tmp_norm_"+scene.name)
                            self._plot_predictions(
                                local_test_batch,
                                names=local_nodes,
                                predictions_torch=local_traj_pred_torch,
                                save_name="./tmp_"+scene.name+"/eval_"
                                + str(scene.name)
                                + "_"
                                + str(t).zfill(3)
                                + "_"
                                + str(s).zfill(3)
                                + "_local.jpg",
                                ade=batch_error_dict[node_type]["ade"][0],
                                fde=batch_error_dict[node_type]["fde"][0],
                                sade=batch_error_dict[node_type]["sade"][0],
                                sfde=batch_error_dict[node_type]["sfde"][0],
                            )
                            self._plot_predictions(
                                local_test_batch,
                                is_vis_st=True,
                                names=local_nodes,
                                predictions_torch=local_traj_pred_torch,
                                save_name="./tmp_norm_"+scene.name+"/eval_st_"
                                + str(scene.name)
                                + "_"
                                + str(t).zfill(3)
                                + "_"
                                + str(s).zfill(3)
                                + "_local.jpg",
                                ade=batch_error_dict[node_type]["ade"][0],
                                fde=batch_error_dict[node_type]["fde"][0],
                                sade=batch_error_dict[node_type]["sade"][0],
                                sfde=batch_error_dict[node_type]["sfde"][0],
                            )
                            all_test_batch.append(local_test_batch)
                            data_items.append((scene.name, t, s))
                # if self.config.save_trajectories:
                #     time_to_node_id_to_preds.update(
                #         self._get_preds(
                #             traj_pred,
                #             nodes,
                #             timesteps_o,
                #             inference_scene.normalized_px,
                #             inference_scene.img_width,
                #             inference_scene.img_height,
                #         )
                #     )
                # del test_batch
                # del nodes
                # del timesteps_o
                # del batch
                # torch.cuda.empty_cache()
                # break # TODO: only if fast test
            if self.config.save_trajectories:
                with open(
                    osp.join(self.model_dir, scene.name + "_predictions.pkl"), "wb"
                ) as f:
                    pickle.dump(time_to_node_id_to_preds, f)
                if self.config.is_eval_hst:
                    with open(
                        osp.join(self.model_dir, scene.name + "_interpolations.pkl"),
                        "wb",
                    ) as f:
                        pickle.dump(time_to_node_id_to_interpolated_infos, f)
            if self.time:
                all_inference_times.extend(inference_times)
                all_num_iters.extend(num_iters)
                inference_times = np.array(inference_times)
                num_iters = np.array(num_iters)
                print("Average inference time (s/timestep):", np.mean(inference_times))
                print("Std dev inference time (s/timestep):", np.std(inference_times))
                print("Max inference time (s/timestep):", np.max(inference_times))
                print("Min inference time (s/timestep):", np.min(inference_times))
                print("Average number of iterations:", np.mean(num_iters))
                print("Std dev number of iterations:", np.std(num_iters))
                print("Max number of iterations:", np.max(num_iters))
                print("Min number of iterations:", np.min(num_iters))

        if self.time:
            all_inference_times = np.array(all_inference_times)
            all_num_iters = np.array(all_num_iters)
            print()
            print(
                "Average inference time for all scenes (s/timestep):",
                np.mean(all_inference_times),
            )
            print(
                "Std dev inference time for all scenes (s/timestep):",
                np.std(all_inference_times),
            )
            print(
                "Max inference time for all scenes (s/timestep):",
                np.max(all_inference_times),
            )
            print(
                "Min inference time for all scenes (s/timestep):",
                np.min(all_inference_times),
            )
            print(
                "Average number of iterations for all scenes (s/timestep):",
                np.mean(all_num_iters),
            )
            print(
                "Std dev number of iterations for all scenes (s/timestep):",
                np.std(all_num_iters),
            )
            print(
                "Max number of iterations for all scenes (s/timestep):",
                np.max(all_num_iters),
            )
            print(
                "Min number of iterations for all scenes (s/timestep):",
                np.min(all_num_iters),
            )
            return None, None, None, None, None

        print(
            "Average number of steps required for solving ODE:",
            sum(num_steps_for_all_samples_list)
            / len(num_steps_for_all_samples_list)
            / self.config.num_samples,
        )

        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)
        kde = np.mean(eval_kde_batch_errors)
        ade_most_likely = np.mean(eval_ade_most_likely_batch_errors)
        fde_most_likely = np.mean(eval_fde_most_likely_batch_errors)
        ade_mean = np.mean(eval_mean_ade_batch_errors)
        fde_mean = np.mean(eval_mean_fde_batch_errors)
        ade_std = np.mean(eval_std_ade_batch_errors)
        fde_std = np.mean(eval_std_fde_batch_errors)
        ade_max = np.max(eval_ade_list_batch_errors)
        fde_max = np.max(eval_fde_list_batch_errors)
        ade_min = np.min(eval_ade_list_batch_errors)
        fde_min = np.min(eval_fde_list_batch_errors)
        if self.config.is_eval_hst:
            ade_one_fourth = np.mean(eval_ade_one_fourth_batch_errors)
            ade_two_fourth = np.mean(eval_ade_two_fourth_batch_errors)
            ade_three_fourth = np.mean(eval_ade_three_fourth_batch_errors)
            ade_most_likely_one_fourth = np.mean(
                eval_ade_most_likely_one_fourth_batch_errors
            )
            ade_most_likely_two_fourth = np.mean(
                eval_ade_most_likely_two_fourth_batch_errors
            )
            ade_most_likely_three_fourth = np.mean(
                eval_ade_most_likely_three_fourth_batch_errors
            )
            ade_mean_one_fourth = np.mean(eval_ade_mean_one_fourth_batch_errors)
            ade_mean_two_fourth = np.mean(eval_ade_mean_two_fourth_batch_errors)
            ade_mean_three_fourth = np.mean(eval_ade_mean_three_fourth_batch_errors)
            kde_one_fourth = np.mean(eval_kde_one_fourth_batch_errors)
            kde_two_fourth = np.mean(eval_kde_two_fourth_batch_errors)
            kde_three_fourth = np.mean(eval_kde_three_fourth_batch_errors)

        sade = np.mean(eval_sade_batch_errors)
        sfde = np.mean(eval_sfde_batch_errors)
        sade_mean = np.mean(eval_mean_sade_batch_errors)
        sfde_mean = np.mean(eval_mean_sfde_batch_errors)
        sade_std = np.mean(eval_std_sade_batch_errors)
        sfde_std = np.mean(eval_std_sfde_batch_errors)

        if self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50

        if sampling == "ddim":
            print(f"Sampling: {sampling} Num of steps: {step}")

        print(
            "Below results optimize ADE rather than OSPA, so the ADE reported here should be less than or equal to the ADE reported by the OSPA evaluation script."
        )
        self.log.info(
            "Below results optimize ADE rather than OSPA, so the ADE reported here should be less than or equal to the ADE reported by the OSPA evaluation script."
        )
        print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde} KDE: {kde}")
        self.log.info(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde} KDE: {kde}")
        print(
            f"Epoch {epoch} Most Likely ADE: {ade_most_likely} Most Likely FDE: {fde_most_likely}"
        )
        self.log.info(
            f"Epoch {epoch} Most Likely ADE: {ade_most_likely} Most Likely FDE: {fde_most_likely}"
        )
        print(
            f"Epoch {epoch} Mean ADE: {ade_mean} (Std: {ade_std} (Min: {ade_min}, Max: {ade_max})) Mean FDE: {fde_mean} (Std: {fde_std} (Min: {fde_min}, Max: {fde_max}))"
        )
        self.log.info(
            f"Epoch {epoch} Mean ADE: {ade_mean} (Std: {ade_std} (Min: {ade_min}, Max: {ade_max})) Mean FDE: {fde_mean} (Std: {fde_std} (Min: {fde_min}, Max: {fde_max}))"
        )
        print(f"Epoch {epoch} Best Of 20: SADE: {sade} SFDE: {sfde}")
        self.log.info(f"Epoch {epoch} Best Of 20: SADE: {sade} SFDE: {sfde}")
        print(
            f"Epoch {epoch} Mean SADE: {sade_mean} (Std: {sade_std}) Mean SFDE: {sfde_mean} (Std: {sfde_std})"
        )
        self.log.info(
            f"Epoch {epoch} Mean SADE: {sade_mean} (Std: {sade_std}) Mean SFDE: {sfde_mean} (Std: {sfde_std})"
        )
        print(
            f"Epoch {epoch} Detected {total_num_det_tracks} tracks out of {total_num_gt_tracks} tracks"
        )
        self.log.info(
            f"Epoch {epoch} Detected {total_num_det_tracks} tracks out of {total_num_gt_tracks} tracks"
        )
        if print_num_samples_with_collisions:
            print(
                f"Rejection sampling fixed {total_num_samples_with_collisions_fixed} samples out of {total_num_samples_with_collisions} samples with collisions"
            )

        if self.config.is_eval_hst:
            print(
                f"Epoch {epoch} Best Of 20: ADE@1/4: {ade_one_fourth} ADE@2/4: {ade_two_fourth} ADE@3/4: {ade_three_fourth}"
            )
            self.log.info(
                f"Epoch {epoch} Best Of 20: ADE@1/4: {ade_one_fourth} ADE@2/4: {ade_two_fourth} ADE@3/4: {ade_three_fourth}"
            )
            print(
                f"Epoch {epoch} Most Likely ADE@1/4: {ade_most_likely_one_fourth} Most Likely ADE@2/4: {ade_most_likely_two_fourth} Most Likely ADE@3/4: {ade_most_likely_three_fourth}"
            )
            self.log.info(
                f"Epoch {epoch} Most Likely ADE@1/4: {ade_most_likely_one_fourth} Most Likely ADE@2/4: {ade_most_likely_two_fourth} Most Likely ADE@3/4: {ade_most_likely_three_fourth}"
            )
            print(
                f"Epoch {epoch} Mean ADE@1/4: {ade_mean_one_fourth} Mean ADE@2/4: {ade_mean_two_fourth} Mean ADE@3/4: {ade_mean_three_fourth}"
            )
            self.log.info(
                f"Epoch {epoch} Mean ADE@1/4: {ade_mean_one_fourth} Mean ADE@2/4: {ade_mean_two_fourth} Mean ADE@3/4: {ade_mean_three_fourth}"
            )
            print(
                f"Epoch {epoch} KDE@1/4: {kde_one_fourth} KDE@2/4: {kde_two_fourth} KDE@3/4: {kde_three_fourth}"
            )
            self.log.info(
                f"Epoch {epoch} KDE@1/4: {ade_one_fourth} KDE@2/4: {kde_two_fourth} KDE@3/4: {kde_three_fourth}"
            )

        # self._plot_displacement_error_histogram(eval_de_list_batch_errors, 0.1)
        if IS_DATALOADER_DEBUG:
            # Get the 50 worst data items with respect to each metric
            worst_ade_idx = np.argpartition(eval_ade_batch_errors, -50)[-50:]
            worst_ade_idx_sorted = worst_ade_idx[np.argsort(eval_ade_batch_errors[worst_ade_idx])[::-1]]
            worst_ade_data_items = [data_items[idx] for idx in worst_ade_idx_sorted]  # list of tuples (scene.name, t, s)
            for i, d in enumerate(worst_ade_data_items):
                time_t = str(d[1]).zfill(3)
                scene_s = str(d[2]).zfill(3)
                if not os.path.isdir("./tmp_"+d[0]+"/worst_ade"):
                    os.makedirs("./tmp_"+d[0]+"/worst_ade")
                if not os.path.isdir("./tmp_norm_"+d[0]+"/worst_ade"):
                    os.makedirs("./tmp_norm_"+d[0]+"/worst_ade")
                shutil.copyfile("./tmp_"+d[0]+"/eval_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_"+d[0]+"/worst_ade/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")
                shutil.copyfile("./tmp_norm_"+d[0]+"/eval_st_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_norm_"+d[0]+"/worst_ade/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")
            worst_fde_idx = np.argpartition(eval_fde_batch_errors, -50)[-50:]
            worst_fde_idx_sorted = worst_fde_idx[np.argsort(eval_fde_batch_errors[worst_fde_idx])[::-1]]
            worst_fde_data_items = [data_items[idx] for idx in worst_fde_idx_sorted]  # list of tuples (scene.name, t, s)
            for i, d in enumerate(worst_fde_data_items):
                time_t = str(d[1]).zfill(3)
                scene_s = str(d[2]).zfill(3)
                if not os.path.isdir("./tmp_"+d[0]+"/worst_fde"):
                    os.makedirs("./tmp_"+d[0]+"/worst_fde")
                if not os.path.isdir("./tmp_norm_"+d[0]+"/worst_fde"):
                    os.makedirs("./tmp_norm_"+d[0]+"/worst_fde")
                shutil.copyfile("./tmp_"+d[0]+"/eval_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_"+d[0]+"/worst_fde/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")
                shutil.copyfile("./tmp_norm_"+d[0]+"/eval_st_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_norm_"+d[0]+"/worst_fde/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")
            worst_sade_idx = np.argpartition(eval_sade_batch_errors, -50)[-50:]
            worst_sade_idx_sorted = worst_sade_idx[np.argsort(eval_sade_batch_errors[worst_sade_idx])[::-1]]
            worst_sade_data_items = [data_items[idx] for idx in worst_sade_idx_sorted]  # list of tuples (scene.name, t, s)
            for i, d in enumerate(worst_sade_data_items):
                time_t = str(d[1]).zfill(3)
                scene_s = str(d[2]).zfill(3)
                if not os.path.isdir("./tmp_"+d[0]+"/worst_sade"):
                    os.makedirs("./tmp_"+d[0]+"/worst_sade")
                if not os.path.isdir("./tmp_norm_"+d[0]+"/worst_sade"):
                    os.makedirs("./tmp_norm_"+d[0]+"/worst_sade")
                shutil.copyfile("./tmp_"+d[0]+"/eval_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_"+d[0]+"/worst_sade/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")
                shutil.copyfile("./tmp_norm_"+d[0]+"/eval_st_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_norm_"+d[0]+"/worst_sade/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")
            worst_sfde_idx = np.argpartition(eval_sfde_batch_errors, -50)[-50:]
            worst_sfde_idx_sorted = worst_sfde_idx[np.argsort(eval_sfde_batch_errors[worst_sfde_idx])[::-1]]
            worst_sfde_data_items = [data_items[idx] for idx in worst_sfde_idx_sorted]  # list of tuples (scene.name, t, s)
            for i, d in enumerate(worst_sfde_data_items):
                time_t = str(d[1]).zfill(3)
                scene_s = str(d[2]).zfill(3)
                if not os.path.isdir("./tmp_"+d[0]+"/worst_sfde"):
                    os.makedirs("./tmp_"+d[0]+"/worst_sfde")
                if not os.path.isdir("./tmp_norm_"+d[0]+"/worst_sfde"):
                    os.makedirs("./tmp_norm_"+d[0]+"/worst_sfde")
                shutil.copyfile("./tmp_"+d[0]+"/eval_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_"+d[0]+"/worst_sfde/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")
                shutil.copyfile("./tmp_norm_"+d[0]+"/eval_st_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg", "./tmp_norm_"+d[0]+"/worst_sfde/eval_"+str(i)+"_"+d[0]+"_"+time_t+"_"+scene_s+"_local.jpg")

        if not self.config.fast_train:
            return ade, fde, kde, ade_most_likely, fde_most_likely, sade, sfde, sade_mean, sfde_mean
        else:
            return ade, fde, kde, ade_mean, fde_mean, sade, sfde, sade_mean, sfde_mean

    def _build(self, init_env=None, num_history=None, prediction_horizon=None):
        self._build_dir()

        self._build_encoder_config(num_history=num_history, prediction_horizon=prediction_horizon)
        self._build_encoder(init_env=init_env)
        self._build_model()
        if not self.sicnav_inference:
            self._build_train_loader()
            self._build_eval_loader()
            self._build_inference_scenes()

            self._build_optimizer()

        print("> Everything built. Have fun :)")

    def _build_dir(self):
        if self.sicnav_inference:
            self.model_dir = osp.join(*self.config.model_path.split("/")[:-1])
        else:
            self.model_dir = osp.join("./experiments", self.config.exp_name)
            self.log_writer = SummaryWriter(log_dir=self.model_dir)
            os.makedirs(self.model_dir, exist_ok=True)
            log_name = "{}.log".format(time.strftime("%Y-%m-%d-%H-%M"))
            if self.test_dataset is not None:
                log_name = f"{self.config.dataset}_{self.test_dataset}_{log_name}"
            else:
                log_name = f"{self.config.dataset}_{log_name}"

            log_dir = osp.join(self.model_dir, log_name)
            self.log = logging.getLogger()
            self.log.setLevel(logging.INFO)
            handler = logging.FileHandler(log_dir)
            handler.setLevel(logging.INFO)
            self.log.addHandler(handler)

            self.log.info("Config:")
            self.log.info(self.config)
            self.log.info("\n")
            self.log.info("Eval on:")
            self.log.info(self.config.dataset)
            self.log.info("\n")

            self.train_data_path = osp.join(
                self.config.data_dir, self.config.dataset + "_train.pkl"
            )
            if self.test_dataset is not None:
                self.eval_env_path = osp.join(
                    self.config.data_dir, self.test_dataset + "_val.pkl"
                )
            else:
                self.eval_env_path = osp.join(
                    self.config.data_dir,
                    self.config.dataset + "_" + self.config.test_split + ".pkl",
                )

            self.inference_data_path = osp.join(
                self.config.data_dir,
                self.config.inference_dataset + "_" + self.config.test_split + ".pkl",
            )

            print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(
            [
                {
                    "params": self.registrar.get_all_but_name_match(
                        "map_encoder"
                    ).parameters()
                },
                {"params": self.model.parameters()},
            ],
            lr=self.config.lr,
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self, num_history=None, prediction_horizon=None):
        self.hyperparams = get_traj_hypers()
        if num_history is not None:
            self.hyperparams["maximum_history_length"] = num_history - 1
            self.hyperparams["prediction_horizon"] = prediction_horizon
        self.hyperparams["enc_rnn_dim_edge"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_edge_influence"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_history"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_future"] = self.config.encoder_dim // 2

        # Use below settings for conditioning MID on robot future motion
        # self.hyperparams["enc_rnn_dim_edge"] = self.config.encoder_dim // 4
        # self.hyperparams["enc_rnn_dim_edge_influence"] = self.config.encoder_dim // 4
        # self.hyperparams["enc_rnn_dim_history"] = self.config.encoder_dim // 4
        # self.hyperparams["enc_rnn_dim_future"] = self.config.encoder_dim // 8

        # registar
        self.registrar = ModelRegistrar(self.model_dir, "cuda" if torch.cuda.is_available() else "cpu")
        if self.sicnav_inference:
            sys.path.insert(0, osp.join(os.getcwd(), "sicnav_diffusion", "JMID", "MID"))
            self.checkpoint = torch.load(self.config.model_path, map_location=self.device)
            self.registrar.load_models(self.checkpoint["encoder"])
        else:
            if self.config.eval_mode:
                epoch = self.config.eval_at
                print(osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))
                self.checkpoint = torch.load(
                    osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"),
                    map_location="cpu",
                )

                self.registrar.load_models(self.checkpoint["encoder"])
            else:
                if self.config.load_chkpt != "None":
                    print(osp.join(self.model_dir, self.config.load_chkpt))
                    self.checkpoint = torch.load(
                        osp.join(self.model_dir, self.config.load_chkpt),
                        map_location="cpu",
                    )

                    self.registrar.load_models(self.checkpoint["encoder"])

            with open(self.train_data_path, "rb") as f:
                self.train_env = dill.load(f, encoding="latin1")
            with open(self.eval_env_path, "rb") as f:
                self.eval_env = dill.load(f, encoding="latin1")

    def _build_encoder(self, init_env=None):
        # update hyper-parameter
        self.hyperparams["maximum_history_length"] = self.config.maximum_history_length
        self.hyperparams["prediction_horizon"] = self.config.prediction_horizon

        self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda" if torch.cuda.is_available() else "cpu")
        if self.sicnav_inference:
            self.encoder.set_environment(init_env)
        else:
            self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()

    def _build_model(self):
        """Define Model"""
        config = self.config
        diffnet = getattr(diffusion, config.diffnet)
        vel_predictor = DiffusionTraj(
            net=diffnet(
                point_dim=2,
                context_dim=config.encoder_dim,
                tf_layer=config.tf_layer,
                residual=False,
            ),
            var_sched=VarianceSchedule(
                num_steps=100, beta_T=5e-2, mode="linear"  ## Diffusion Step
            ),
        )
        model = AutoEncoder(config, encoder=self.encoder, vel_predictor=vel_predictor)
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        if self.config.eval_mode:
            self.model.load_state_dict(self.checkpoint["ddpm"])
        else:
            if self.config.load_chkpt != "None":
                self.model.load_state_dict(self.checkpoint["ddpm"])
                print("> Loaded checkpoint:", self.config.load_chkpt)

        print("> Model built!")

    def _build_train_loader(self):
        if self.config.eval_mode:
            return None

        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, "rb") as f:
            train_env = dill.load(f, encoding="latin1")

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                " "
            )
            train_env.attention_radius[(node_type1, node_type2)] = float(
                attention_radius
            )

        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = (
            self.train_env.scenes_freq_mult_prop
            if config.scene_freq_mult_train
            else None
        )

        train_index_path = osp.join(
            self.config.data_dir, self.config.dataset + "_train_index.pkl"
        )
        self.train_dataset = EnvironmentDataset(
            train_env,
            self.hyperparams["state"],
            self.hyperparams["pred_state"],
            scene_freq_mult=self.hyperparams["scene_freq_mult_train"],
            node_freq_mult=self.hyperparams["node_freq_mult_train"],
            hyperparams=self.hyperparams,
            min_history_timesteps=1,
            min_future_timesteps=self.hyperparams["prediction_horizon"],
            return_robot=not self.hyperparams["incl_robot_node"],
            is_joint_pred=self.is_joint_pred,
            regenerate_index=config.regenerate_indexes,
            index_path=train_index_path,
            is_dataloader_debug=IS_DATALOADER_DEBUG,
        )

        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            # Only predict pedestrians. This will still use all nodes as neighbours though, as desired
            if node_type_data_set.node_type != "PEDESTRIAN":
                continue
            node_type_dataloader = utils.data.DataLoader(
                node_type_data_set,
                collate_fn=joint_pred_train_collate if self.is_joint_pred else collate,
                pin_memory=True,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.preprocess_workers,
            )
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    def _build_eval_loader_sicnav_inference(self, eval_env):
        config = self.config
        self.eval_env = eval_env

        for attention_radius_override in config.override_attention_radius:
            (
                node_type1,
                node_type2,
                attention_radius,
            ) = attention_radius_override.split(" ")
            self.eval_env.attention_radius[(node_type1, node_type2)] = float(
                attention_radius
            )

        if self.eval_env.robot_type is None and self.hyperparams["incl_robot_node"]:
            self.eval_env.robot_type = self.eval_env.NodeType[
                0
            ]  # TODO: Make more general, allow the user to specify?
            for scene in self.eval_env.scenes:
                scene.add_robot_from_nodes(self.eval_env.robot_type)
        self.eval_scenes = self.eval_env.scenes

    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []

        if config.eval_every is not None:
            with open(self.eval_env_path, "rb") as f:
                self.eval_env = dill.load(f, encoding="latin1")

            for attention_radius_override in config.override_attention_radius:
                (
                    node_type1,
                    node_type2,
                    attention_radius,
                ) = attention_radius_override.split(" ")
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(
                    attention_radius
                )

            if self.eval_env.robot_type is None and self.hyperparams["incl_robot_node"]:
                self.eval_env.robot_type = self.eval_env.NodeType[
                    0
                ]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes

            if self.config.eval_mode:
                # Skip building the validation data loader
                return None

            if self.test_dataset is not None:
                eval_index_path = osp.join(
                    self.config.data_dir, self.test_dataset + "_val_index.pkl"
                )
            else:
                eval_index_path = osp.join(
                    self.config.data_dir,
                    self.config.dataset + "_" + self.config.test_split + "_index.pkl",
                )
            self.eval_dataset = EnvironmentDataset(
                self.eval_env,
                self.hyperparams["state"],
                self.hyperparams["pred_state"],
                scene_freq_mult=self.hyperparams["scene_freq_mult_eval"],
                node_freq_mult=self.hyperparams["node_freq_mult_eval"],
                hyperparams=self.hyperparams,
                min_history_timesteps=self.hyperparams["minimum_history_length"],
                min_future_timesteps=self.hyperparams["prediction_horizon"]
                if self.config.test_split != "test"
                else 0,
                return_robot=not self.hyperparams["incl_robot_node"],
                is_joint_pred=self.is_joint_pred,
                regenerate_index=config.regenerate_indexes,
                index_path=eval_index_path,
            )
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                # Only predict pedestrians. This will still use all nodes as neighbours though
                if node_type_data_set.node_type != "PEDESTRIAN":
                    continue
                node_type_dataloader = utils.data.DataLoader(
                    node_type_data_set,
                    collate_fn=joint_pred_train_collate if self.is_joint_pred else collate,
                    pin_memory=True,
                    batch_size=config.eval_batch_size,
                    shuffle=True,
                    num_workers=config.preprocess_workers,
                )
                self.eval_data_loader[
                    node_type_data_set.node_type
                ] = node_type_dataloader

        print("> Dataset built!")

    def _build_inference_scenes(self):
        config = self.config
        self.inference_scenes = []

        with open(self.inference_data_path, "rb") as f:
            self.inference_env = dill.load(f, encoding="latin1")

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                " "
            )
            self.inference_env.attention_radius[(node_type1, node_type2)] = float(
                attention_radius
            )

        if (
            self.inference_env.robot_type is None
            and self.hyperparams["incl_robot_node"]
        ):
            self.inference_env.robot_type = self.inference_env.NodeType[
                0
            ]  # TODO: Make more general, allow the user to specify?
            for scene in self.inference_env.scenes:
                scene.add_robot_from_nodes(self.inference_env.robot_type)

        self.inference_scenes = self.inference_env.scenes

    def _build_offline_scene_graph(self):
        if self.hyperparams["offline_scene_graph"] == "yes":
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(
                    self.train_env.attention_radius,
                    self.hyperparams["edge_addition_filter"],
                    self.hyperparams["edge_removal_filter"],
                )
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(
                    self.eval_env.attention_radius,
                    self.hyperparams["edge_addition_filter"],
                    self.hyperparams["edge_removal_filter"],
                )
                print(f"Created Scene Graph for Evaluation Scene {i}")

    def _save_model(self, epoch):
        checkpoint = {
            "encoder": self.registrar.model_dict,
            "ddpm": self.model.state_dict(),  # TODO (alem): Change from "ddpm" to something that generalizes for CNF
        }
        torch.save(
            checkpoint,
            osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"),
        )

    def _save_model_if_best(self, ade, best_val_ade, epoch):
        """Returns the best validation ADE and if the model was saved or not"""
        if ade > best_val_ade:
            return best_val_ade, False
        checkpoint = {
            "encoder": self.registrar.model_dict,
            "ddpm": self.model.state_dict(),  # TODO (alem): Change from "ddpm" to something that generalizes for CNF
        }
        torch.save(
            checkpoint,
            osp.join(self.model_dir, f"{self.config.dataset}_best_epoch{epoch}.pt"),
        )
        return ade, True

    def _get_inference_scene(self, scene):
        # Assume scenes have unique names
        for i_scene in self.inference_scenes:
            if i_scene.name == scene.name:
                return i_scene
        return None

    def _compute_all_ades(
        self, nodes_list_1, nodes_list_2, normalized_px, img_width, img_height
    ):
        """Computes the ade between all pairs of nodes between nodes_list_1 and nodes_list_2"""
        ade_matrix = np.full(
            (len(nodes_list_1), len(nodes_list_2)), np.finfo(np.float64).max
        )
        for i, node_1 in enumerate(nodes_list_1):
            for j, node_2 in enumerate(nodes_list_2):
                # Assume that timesteps are consistent across all nodes
                ts = node_1.get_overlapping_timesteps(node_2)
                if len(ts) == 0:
                    continue
                x_y_positions_1 = node_1.get_x_and_y(ts)
                x_y_positions_1 = self._un_preprocess_data(
                    x_y_positions_1, node_1, normalized_px, img_width, img_height
                )
                x_y_positions_2 = node_2.get_x_and_y(ts)
                x_y_positions_2 = self._un_preprocess_data(
                    x_y_positions_2, node_2, normalized_px, img_width, img_height
                )
                # TODO(f-sato)
                ade_matrix[i, j] = evaluation.compute_ade(
                    x_y_positions_1,
                    x_y_positions_2,
                    interpolated_future=np.full(
                        (x_y_positions_1.shape[0]), fill_value=False
                    ),
                )  # TODO(alem): Should be calculated in pixels
        return ade_matrix

    def _postprocess_optimal_indices(self, row_ind, col_ind, ade_matrix):
        """Remove matches with an ADE of the max value a float can hold since these are not actual matches. The max value a float can hold is meant to indicate there is no `edge.`"""
        adjusted_row_ind = []
        adjusted_col_ind = []
        for i in range(len(row_ind)):
            if ade_matrix[row_ind[i], col_ind[i]] > np.finfo(np.float64).max / 10:
                continue
            adjusted_row_ind.append(row_ind[i])
            adjusted_col_ind.append(col_ind[i])
        return np.array(adjusted_row_ind), np.array(adjusted_col_ind)

    def _get_optimal_node_matches(self, ade_matrix):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            ade_matrix
        )  # nodes, other_nodes
        row_ind, col_ind = self._postprocess_optimal_indices(
            row_ind, col_ind, ade_matrix
        )
        return row_ind, col_ind

    def _associate_nodes_with_other_nodes(
        self, nodes, other_nodes, normalized_px, img_width, img_height
    ):
        """Associates each node in `nodes` with a unique node in `other_nodes`. Each node can only be associated once.
        Nodes are associated such that the total ADE of all associations is minimized. Returns an array of nodes indices
        and an array of other_nodes indices defining the node associations (pairs)."""
        ade_matrix = self._compute_all_ades(
            nodes, other_nodes, normalized_px, img_width, img_height
        )
        row_ind, col_ind = self._get_optimal_node_matches(ade_matrix)
        return row_ind, col_ind

    def _add_mean(self, data, node):
        data = data + np.expand_dims(node.get_mean_x_and_y(), axis=0)
        return data

    def _unnormalize_pixels(self, data, normalized_px, img_width=0, img_height=0):
        if not normalized_px:
            return data
        data = data * np.array([[img_width, img_height]])
        return data

    def _un_preprocess_data(self, data, node, normalized_px, img_width=0, img_height=0):
        """Data (timesteps x 2). Assume second dimension of data is x-position then y-position. `node` is the node data came
        from. `normalized_px` is a boolean representing if the data is normalized pixel values or not.
        """
        data = self._add_mean(data, node)  # Data is mean-centered (see process_data.py)
        data = self._unnormalize_pixels(data, normalized_px, img_width, img_height)
        return data

    def exists_at_t(self, gt_node, ts):
        _, paddingl, paddingu = gt_node.scene_ts_to_node_ts([ts, ts])
        return not ((paddingl > 0) or (paddingu > 0))


    def _get_preds_sicnav_inference(
        self, traj_pred, nodes, timesteps, normalized_px, img_width, img_height
    ):
        """
        Return predictions in the odometry frame as Numpy arrays.

        Inputs:
            traj_pred (torch.tensor): self.num_samples x B x pred_horiz x 2
        Outputs:
            predictions (torch.tensor: self.num_samples x B x pred_horiz x 2)
        """
        predictions = self._un_preprocess_data(traj_pred, nodes[0], normalized_px, img_width, img_height)  # Assume mean for all nodes is the same
        sort_ind = torch.tensor([int(n.id) for n in nodes], device=self.device).argsort(dim=0)
        predictions = predictions[:,sort_ind]
        # Assume we only make a prediction for one timestep (all entries in timesteps should be the same). This means "B" should be the number of agents in the scene
        # sorted_predictions = torch.zeros((traj_pred.size(1), traj_pred.size(0), traj_pred.size(2), traj_pred.size(3)), device=self.device)  # num_hum x samples x 12 x 2
        # for i in range(len(timesteps)):
        #     predictions = torch.permute(traj_pred[:, [i]], (1, 0, 2, 3))
        #     predictions_shape = predictions.shape
        #     predictions = torch.reshape(predictions, (-1, 2))
        #     predictions = self._un_preprocess_data(
        #         predictions, nodes[i], normalized_px, img_width, img_height
        #     )
        #     predictions = torch.reshape(predictions, predictions_shape)
        #     sorted_predictions[int(nodes[i].id)] = predictions[0, :, :, :]
        # return sorted_predictions
        return predictions

    def _get_preds(
        self, traj_pred, nodes, timesteps, normalized_px, img_width, img_height
    ):
        """
        Return predictions in the odometry frame
        traj_pred (np.array): Samples x B x 12 x 2
        """
        predictions_dict = {}
        for i, ts in enumerate(timesteps):
            if ts not in predictions_dict.keys():
                predictions_dict[ts] = dict()
            predictions = np.transpose(traj_pred[:, [i]], (1, 0, 2, 3))
            predictions_shape = predictions.shape
            predictions = np.reshape(predictions, (-1, 2))
            predictions = self._un_preprocess_data(
                predictions, nodes[i], normalized_px, img_width, img_height
            )
            predictions = np.reshape(predictions, predictions_shape)
            predictions_dict[ts][int(nodes[i].id)] = predictions[0, :, :, :]
        return predictions_dict

    def _plot_predictions(self, batch, names=None, save_name="tmp.jpg", predictions=None, predictions_torch=None, is_vis_st=False, ade=None, fde=None, sade=None, sfde=None):
        raise NotImplementedError

    def _dump_mask(self, attn_mask, loss_mask):
        pass

    def _plot_displacement_error_histogram(self, de_list, bin_interval=0.1):
        """`de_list` is a list of the displaement erorrs. `bin_interval` is the width of each bin in meters"""
        plt.figure(figsize=(10, 8))
        plt.hist(
            de_list,
            bins=np.arange(min(de_list), max(de_list) + bin_interval, bin_interval),
        )
        plt.xlabel("Displacement Error (m)")
        plt.ylabel("Number of Errors")
        plt.title(
            "Histogram of Number of Errors per Displacement Error Binned in Intervals of 0.1m"
        )
        plt.xticks(ticks=range(0, math.ceil(max(de_list))))
        plt.tight_layout()
        plt.savefig(osp.join(self.model_dir, "de_distribution.jpg"))

    def _reshape_batch_for_bs_one(self, batch):
        for i in range(len(batch)):
            if type(batch[i]) is dict:
                for key in batch[i].keys():
                    tmp_element = copy.copy(batch[i][tuple(key)][0])
                    batch[i][tuple(key)] = tmp_element
            else:
                if torch.is_tensor(batch[i]):
                    batch[i] = batch[i][0]
        return batch
