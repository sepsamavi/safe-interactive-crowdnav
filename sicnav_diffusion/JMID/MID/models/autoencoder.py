from torch.nn import Module

from .encoders.model_utils import ModeKeys

class AutoEncoder(Module):
    def __init__(self, config, encoder, vel_predictor):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.vel_predictor = vel_predictor
        # self.diffusion = vel_predictor

    def encode(self, mode, batch, node_type):
        z = self.encoder.get_latent(mode, batch, node_type)
        return z

    def generate_sicnav_inference(
        self,
        batch,
        node_type,
        num_points,
        sample,
        bestof,
        flexibility=0.0,
        ret_traj=False,
        sampling="ddpm",
        step=100,
        with_constraints=True,
    ):
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(ModeKeys.PREDICT, batch, node_type)
        predicted_y_vel, num_steps_for_all_samples = self.vel_predictor.sample_sicnav_inference(
        # predicted_y_vel, num_steps_for_all_samples = self.diffusion.sample(
            num_points,
            encoded_x,
            sample,
            bestof,
            flexibility=flexibility,
            ret_traj=ret_traj,
            sampling=sampling,
            step=step,
            with_constraints=with_constraints,
            dynamics=dynamics,
        )
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        # return predicted_y_pos.cpu().detach().numpy(), num_steps_for_all_samples
        return predicted_y_pos, num_steps_for_all_samples



    def generate(
        self,
        batch,
        node_type,
        num_points,
        sample,
        bestof,
        flexibility=0.0,
        ret_traj=False,
        sampling="ddpm",
        step=100,
        with_constraints=True,
        constraint_type=None,
        as_numpy=True,
    ):
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(ModeKeys.PREDICT, batch, node_type)
        (
            predicted_y_vel,
            num_steps_for_all_samples,
            num_iter,
            num_samples_with_collisions,
            num_samples_with_collisions_fixed,
        ) = self.vel_predictor.sample(
            num_points,
            encoded_x,
            sample,
            bestof,
            flexibility=flexibility,
            ret_traj=ret_traj,
            sampling=sampling,
            step=step,
            with_constraints=with_constraints,
            constraint_type=constraint_type,
            dynamics=dynamics,
        )
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        if as_numpy:
            return (
                predicted_y_pos.cpu().detach().numpy(),
                num_steps_for_all_samples,
                num_iter,
                num_samples_with_collisions,
                num_samples_with_collisions_fixed,
            )
        else:
            return (
                predicted_y_pos,
                num_steps_for_all_samples,
                num_iter,
                num_samples_with_collisions,
                num_samples_with_collisions_fixed,
            )

    def get_loss(self, mode, batch, node_type, batch_size=None, attn_mask=None, loss_mask=None):
        (
            first_history_index,
            x_t,
            y_t,
            x_st_t,
            y_st_t,
            neighbors_data_st,
            neighbors_edge_value,
            robot_traj_st_t,
            map,
            _,
        ) = batch

        feat_x_encoded = self.encode(mode, batch, node_type)  # B * 64
        loss = self.vel_predictor.get_loss(y_t.cuda(), feat_x_encoded, batch_size=batch_size, attn_mask=attn_mask, loss_mask=loss_mask)
        # loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        return loss
