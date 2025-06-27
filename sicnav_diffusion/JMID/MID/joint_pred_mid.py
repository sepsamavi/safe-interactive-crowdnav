import numpy as np
from mid import MID
from models.autoencoder import AutoEncoder
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj, VarianceSchedule
from matplotlib import pyplot as plt
import distinctipy
import evaluation
import pandas as pd
from dataset.preprocessing import remove_nan_rows, allclose_with_nan, check_padding_value

human_color = "gray"
hist_color = "orange"
gt_fut_color = "red"
# pred_color = "tab:green"
cyclist_hist_color = "yellow"
robot_hist_color = "blue"
robot_color = "yellow"

human_radius = 0.15
robot_radius = 0.25

NUM_VISUALIZING_SAMPLES = 20
IS_FIXED_AXIS = False  # True
X_AXIS = [-5, 5]
Y_AXIS = [-5, 5]


class JointPredMID(MID):
    def __init__(
        self,
        config,
        test_dataset=None,
        is_joint_pred=True,
        time=False,
    ):
        is_joint_pred = True
        super().__init__(config, test_dataset, is_joint_pred, time)

    def _plot_predictions(
        self,
        test_batch,
        names=None,
        predictions_torch=None,
        is_vis_st=False,
        save_name="tmp.jpg",
        ade=None,
        fde=None,
        sade=None,
        sfde=None,
    ):
        if predictions_torch is not None:
            predictions = predictions_torch.cpu().detach().numpy()
        else:
            predictions = None

        # Get target agent's position normalized by attention radius
        base_st_target_current_pos = np.expand_dims(test_batch[1].cpu().detach().numpy()[0, -1, 0:2] / self.train_env.attention_radius[("PEDESTRIAN", "PEDESTRIAN")],
                                                    axis=0)
        if is_vis_st is False:
            pedestrian_hists = test_batch[1].cpu().detach().numpy()
            futs_vel = test_batch[2].cpu().detach().numpy()
            current_pos = np.expand_dims(pedestrian_hists[:, -1, 0:2], axis=1)
            futs = (
                np.cumsum(futs_vel, axis=1)
                * self.encoder.node_models_dict["PEDESTRIAN"].dynamic.dt
                + current_pos
            )
            # test code
            gt_futs_pos = test_batch[9].cpu().detach().numpy()
            assert allclose_with_nan(futs, gt_futs_pos), "Something wrong to change velocity to position for future trajectory."
        else:
            # Since the position of neighboring pedestrians is normalized for each pedestrians,
            # visualizing the normalized trajectory for each pedestrian will result in a large number of visualized images, which will be difficult to see.
            # So I only visualize input normalized for the target pedestrian for now.
            pedestrian_hists = test_batch[3].cpu().detach().numpy()
            neighbor_hists = test_batch[5][("PEDESTRIAN", "PEDESTRIAN")]
            # Add neigborhood information
            for i, neighbor_hist_for_target_agent in enumerate(neighbor_hists[0]):
                neighbor_hist_for_target_agent = check_padding_value(neighbor_hist_for_target_agent, base_st_target_current_pos)
                pedestrian_hists[i+1,:] = neighbor_hist_for_target_agent


            futs_vel = test_batch[4].cpu().detach().numpy()
            current_pos = np.expand_dims(pedestrian_hists[:, -1, 0:2], axis=1)
            std = [
                self.train_env.standardization["PEDESTRIAN"]["velocity"]["x"]["std"],
                self.train_env.standardization["PEDESTRIAN"]["velocity"]["y"]["std"],
            ]
            std = np.stack(std).reshape(1, 1, 2)
            futs_vel = (
                futs_vel
                * std
                / self.train_env.attention_radius[("PEDESTRIAN", "PEDESTRIAN")]
            )
            futs = (
                np.cumsum(futs_vel, axis=1)
                * self.encoder.node_models_dict["PEDESTRIAN"].dynamic.dt
                + current_pos
            )

        target_current_pos = (
            test_batch[1].cpu().detach().numpy()[0, -1, 0:2].reshape(1, -1)
        )
        target_current_pos = np.tile(
            target_current_pos,
            (pedestrian_hists.shape[1], 1),
        )
        neighbor_hists = test_batch[5]
        robot_hists = []
        cyclist_hists = []
        if isinstance(neighbor_hists, dict):
            for edge_type in neighbor_hists.keys():
                if "ROBOT" in edge_type[1].name:
                    for robot_hist in neighbor_hists[edge_type][0]:
                        if is_vis_st:
                            robot_hist = robot_hist.cpu().detach().numpy()[:, 0:2]
                            robot_hist = check_padding_value(robot_hist, base_st_target_current_pos)
                        else:
                            robot_hist = robot_hist.cpu().detach().numpy()[:, 0:2]
                            robot_hist = check_padding_value(robot_hist, base_st_target_current_pos)
                            robot_hist = (
                                robot_hist
                                * self.train_env.attention_radius[
                                    ("PEDESTRIAN", "JRDB_ROBOT")
                                ]
                            ) + target_current_pos
                        robot_hists.append(robot_hist)

                elif "BICYCLE" in edge_type[1].name:
                    for cyclist_hist in neighbor_hists[edge_type][0]:
                        if is_vis_st:
                            cyclist_hist = cyclist_hist.cpu().detach().numpy()[:, 0:2]
                            cyclist_hist = check_padding_value(cyclist_hist, base_st_target_current_pos)
                        else:
                            cyclist_hist = cyclist_hist.cpu().detach().numpy()[:, 0:2]
                            cyclist_hist = check_padding_value(cyclist_hist, base_st_target_current_pos)
                            cyclist_hist = (
                                cyclist_hist[:, :2]
                                * self.train_env.attention_radius[
                                    ("PEDESTRIAN", "BICYCLE")
                                ]
                            ) + target_current_pos
                        cyclist_hists.append(cyclist_hist)

        plt.close("all")
        cond_2_plots = (
            predictions is not None and NUM_VISUALIZING_SAMPLES < predictions.shape[0]
        )
        if cond_2_plots:
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            # fig_perstep, axs_perstep = plt.subplots(predictions_torch.size(1), predictions_torch.size(2), figsize=(2+2*predictions_torch.size(2), 1+predictions_torch.size(1)*2))
            # if predictions_torch.size(1) == 1:
            #     axs_perstep = [axs_perstep]
            # for h_idx in range(predictions_torch.size(1)):
            #     for s_idx in range(predictions_torch.size(2)):
            #         axs_perstep[h_idx][s_idx].axis("equal")
            #         axs_perstep[h_idx][s_idx].grid(True)
            #         axs_perstep[h_idx][s_idx].scatter(predictions[:, h_idx, s_idx, 0], predictions[:, h_idx, s_idx, 1], c='gray', s=1)

            # fig_perstep.savefig(save_name.replace('.jpg', '_perstep.jpg'))

            # ax = axs[0]
        else:
            fig, ax0 = plt.subplots(figsize=(10, 10))
            axs = [ax0]

        pred_colors = distinctipy.get_colors(NUM_VISUALIZING_SAMPLES)
        if cond_2_plots:
            # get the top_k_predictions
            top_k_predictions_torch, _ = evaluation.evaluation.get_most_likely_samples(
                predictions_torch, self, NUM_VISUALIZING_SAMPLES
            )
            top_k_predictions = top_k_predictions_torch.cpu().detach().numpy()
        for a_idx in range(pedestrian_hists.shape[0]):
            hist = pedestrian_hists[a_idx, :, :2]
            fut = futs[a_idx, :, :]

            # remove nan for filled nan in data loader
            hist = remove_nan_rows(hist)
            fut = remove_nan_rows(fut)

            if predictions is not None:
                if is_vis_st is False:
                    agent_preds = predictions[:, a_idx]
                    if cond_2_plots:
                        top_k_agent_preds = top_k_predictions[a_idx]
                else:
                    target_current_pos = (
                        test_batch[1]
                        .cpu()
                        .detach()
                        .numpy()[0, -1, 0:2]
                        .reshape(1, 1, -1)
                    )
                    target_current_pos_for_all_samples = np.tile(
                        target_current_pos,
                        (predictions.shape[0], predictions.shape[2], 1),
                    )
                    agent_preds = (
                        predictions[:, a_idx] - target_current_pos_for_all_samples
                    ) / self.train_env.attention_radius[("PEDESTRIAN", "PEDESTRIAN")]
                    if cond_2_plots:
                        target_current_pos_for_top_k = np.tile(
                            target_current_pos,
                            (NUM_VISUALIZING_SAMPLES, predictions.shape[2], 1),
                        )
                        top_k_agent_preds = (
                            top_k_predictions[a_idx] - target_current_pos_for_top_k
                        ) / self.train_env.attention_radius[
                            ("PEDESTRIAN", "PEDESTRIAN")
                        ]

            # hist line style should be -- for all agents
            linewidth = 4 if a_idx == 0 else 2
            label = "Hist." if a_idx == 0 else None
            for ax in axs:
                ax.plot(
                    hist[:, 0],
                    hist[:, 1],
                    color=hist_color,
                    linestyle="-",
                    label=label,
                    linewidth=linewidth,
                )
            if predictions is not None:
                label = "Pred." if a_idx == 0 else None
                # for i in range(agent_preds.shape[0]):
                if cond_2_plots:
                    for i in range(agent_preds.shape[0]):
                        axs[0].plot(
                            agent_preds[i, :, 0],
                            agent_preds[i, :, 1],
                            color="gray",
                            linestyle="-",
                            marker="x",
                            markersize=4,
                            label=label,
                            linewidth=1,
                            alpha=0.2,
                        )
                        if i < NUM_VISUALIZING_SAMPLES:
                            axs[1].plot(
                                top_k_agent_preds[i, :, 0],
                                top_k_agent_preds[i, :, 1],
                                color=pred_colors[i],
                                linestyle="-",
                                marker="x",
                                markersize=4,
                                label=label,
                                linewidth=1,
                                alpha=0.75,
                            )
                else:
                    for i in range(NUM_VISUALIZING_SAMPLES):
                        axs[0].plot(
                            agent_preds[i, :, 0],
                            agent_preds[i, :, 1],
                            color=pred_colors[i],
                            linestyle="-",
                            marker="x",
                            markersize=4,
                            label=label,
                            linewidth=1,
                            alpha=0.75,
                        )

            label = "GT Fut." if a_idx == 0 else None
            for ax in axs:
                ax.plot(
                    fut[:, 0],
                    fut[:, 1],
                    color=gt_fut_color,
                    linestyle="-",
                    label=label,
                    marker=".",
                    markersize=2,
                    linewidth=linewidth * 0.5,
                )
            if names is not None:
                for ax in axs:
                    circle = plt.Circle(
                        (hist[-1, 0], hist[-1, 1]),
                        human_radius if is_vis_st is False else human_radius / self.train_env.attention_radius[("PEDESTRIAN", "PEDESTRIAN")],
                        color=human_color,
                        fill=True,
                        alpha=0.3,
                    )
                    ax.add_artist(circle)
                    # make a text beside the center of the circle with the agent name
                    plt.text(
                        hist[-1, 0],
                        hist[-1, 1],
                        str(names[a_idx].id),
                        fontsize=6,
                        color="black",
                    )

        # fig.savefig(save_name)

        # visualize robot and cyclist history
        # if is_vis_st is True:
        for a_idx in range(len(cyclist_hists)):
            cyclist_hist = cyclist_hists[a_idx]
            # remove nan for filled nan in data loader
            cyclist_hist = remove_nan_rows(cyclist_hist)
            label = "BICYCLE Hist."
            for ax in axs:
                ax.plot(
                    cyclist_hist[:, 0],
                    cyclist_hist[:, 1],
                    color=cyclist_hist_color,
                    linestyle="-",
                    label=label,
                    linewidth=2,
                )
        label = "Robot Hist."
        for a_idx in range(len(robot_hists)):
            robot_hist = robot_hists[a_idx]
            # remove nan for filled nan in data loader
            robot_hist = remove_nan_rows(robot_hist)
            for ax in axs:
                ax.plot(
                    robot_hist[:, 0],
                    robot_hist[:, 1],
                    color=robot_hist_color,
                    linestyle="-",
                    label=label,
                    linewidth=2,
                    marker="o",
                    markersize=1,
                )
                rob_circle = plt.Circle(
                    (robot_hist[-1, 0], robot_hist[-1, 1]),
                    robot_radius if is_vis_st is False else robot_radius / self.train_env.attention_radius[("PEDESTRIAN", "PEDESTRIAN")],
                    color=robot_color,
                    fill=True,
                    alpha=0.3,
                )
                ax.add_artist(rob_circle)

        # Delete overlapped legends
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax.legend(unique_labels.values(), unique_labels.keys())

            if IS_FIXED_AXIS:
                ax.set_xlim(X_AXIS[0], X_AXIS[1])
                ax.set_ylim(Y_AXIS[0], Y_AXIS[1])
            else:
                ax.axis("equal")

        # Display min of 20 ADE, min of 20 FDE, min of 20 SADE, min of 20 SFDE
        plt.figtext(0.5, 0.01, f"ADE: {round(ade, 2)}, FDE: {round(fde, 2)}, SADE: {round(sade, 2)}, SFDE: {round(sfde, 2)}", ha="center", fontsize=18)
        fig.savefig(save_name, dpi=300)

    def _dump_mask(self, attn_mask, loss_mask):
        attn_mask_df = pd.DataFrame(attn_mask.numpy())
        num_agents = int(loss_mask.shape[1] / self.hyperparams["prediction_horizon"])
        labels = [f"B{b}_A{a}_T{t}" for b in range(self.config["batch_size"]) for a in range(num_agents) for t in range(self.hyperparams["prediction_horizon"])]
        attn_mask_df.to_csv('attn_mask.csv', header=labels)

        loss_mask_df = pd.DataFrame(loss_mask.numpy())
        labels = [f"A{a}_T{t}" for a in range(num_agents) for t in range(self.hyperparams["prediction_horizon"])]
        loss_mask_df.to_csv('loss_mask.csv', header=labels)
