import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb

FUTURE_LENGTH = 8


class VarianceSchedule(Module):
    def __init__(
        self, num_steps, mode="linear", beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3
    ):
        super().__init__()
        assert mode in ("linear", "cosine")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "cosine":
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[
                i
            ]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (
            1 - flexibility
        )
        return sigmas


class TrajNet(Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList(
            [
                ConcatSquashLinear(2, 128, context_dim + 3),
                ConcatSquashLinear(128, 256, context_dim + 3),
                ConcatSquashLinear(256, 512, context_dim + 3),
                ConcatSquashLinear(512, 256, context_dim + 3),
                ConcatSquashLinear(256, 128, context_dim + 3),
                ConcatSquashLinear(128, 2, context_dim + 3),
            ]
        )

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        out = x
        # pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(
            d_model=2 * context_dim, dropout=0.1, max_len=24
        )
        self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim, nhead=4, dim_feedforward=4 * context_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=tf_layer
        )
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(
            context_dim, context_dim // 2, context_dim + 3
        )
        self.linear = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)
        # self.linear = nn.Linear(128,2)

    def forward(self, x_context, beta, **kwargs):
        x, context = x_context
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class JointPredictionTransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(
            d_model=2 * context_dim, dropout=0.1, max_len=24
        )
        self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim, nhead=4, dim_feedforward=4 * context_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=tf_layer
        )
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(
            context_dim, context_dim // 2, context_dim + 3
        )
        self.linear = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)

    def forward(self, x_context, beta, batch_size=None, mask=None):
        x, context = x_context
        total_agent_num = x.size(0)
        beta = beta.view(total_agent_num, 1, 1)  # (B, 1, 1)
        context = context.view(total_agent_num, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)
        if mask is not None:
            # time x a x dim
            pre_trans = final_emb.permute(1, 0, 2) # a x time x dim
            pre_trans = pre_trans.reshape(
                final_emb.shape[0] * final_emb.shape[1], 1, final_emb.shape[2]
            )
            trans = self.transformer_encoder(src=pre_trans, mask=mask)
            trans = trans.reshape(
                final_emb.shape[1], final_emb.shape[0], final_emb.shape[2]
            )
        else:
            pre_trans = final_emb.reshape(
                final_emb.shape[0] * final_emb.shape[1], 1, final_emb.shape[2]
            )
            trans = self.transformer_encoder(src=pre_trans)
            # trans = pre_trans
            trans = trans.reshape(
               final_emb.shape[0], final_emb.shape[1], final_emb.shape[2])
            trans = trans.permute(1, 0, 2)
            # flag = torch.equal(final_emb.permute(1, 0, 2), trans)

        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

class JointPredictionInstanceTransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(
            d_model=2 * context_dim, dropout=0.1, max_len=24
        )
        self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim * FUTURE_LENGTH,
            nhead=4,
            dim_feedforward=4 * context_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=tf_layer
        )
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(
            context_dim, context_dim // 2, context_dim + 3
        )
        self.linear = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)

    def forward(self, x_context, beta):
        x, context = x_context
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)
        pre_trans = final_emb.permute(1, 0, 2)
        pre_trans = pre_trans.reshape(
            final_emb.shape[1], 1, final_emb.shape[0] * final_emb.shape[2]
        )
        trans = self.transformer_encoder(pre_trans)
        trans = trans.reshape(
            final_emb.shape[1], final_emb.shape[0], final_emb.shape[2]
        )
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class JointPredictionInstanceTransformerConcatLinearv2(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(
            d_model=2 * context_dim, dropout=0.1, max_len=24
        )
        self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim * FUTURE_LENGTH,
            nhead=4,
            dim_feedforward=4 * context_dim,
        )
        self.mlp1 = MLP(
            2 * context_dim * FUTURE_LENGTH, 2 * context_dim * FUTURE_LENGTH
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=tf_layer
        )
        # self.mlp2 = MLP(2 * context_dim * FUTURE_LENGTH, 2 * context_dim * FUTURE_LENGTH)
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(
            context_dim, context_dim // 2, context_dim + 3
        )
        self.linear = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)

    def forward(self, x_context, beta):
        x, context = x_context
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)
        pre_trans = final_emb.permute(1, 0, 2)
        pre_trans = pre_trans.reshape(
            final_emb.shape[1], final_emb.shape[0] * final_emb.shape[2]
        )
        pre_trans = self.mlp1(pre_trans)
        pre_trans = pre_trans.reshape(
            final_emb.shape[1], 1, final_emb.shape[0] * final_emb.shape[2]
        )
        trans = self.transformer_encoder(pre_trans)
        trans = trans.reshape(
            final_emb.shape[1], final_emb.shape[0], final_emb.shape[2]
        )
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class JointPredictionInstanceTransformerConcatLinearv3(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(
            d_model=2 * context_dim, dropout=0.1, max_len=24
        )
        self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim * FUTURE_LENGTH,
            nhead=4,
            dim_feedforward=4 * context_dim,
        )
        self.mlp1 = MLP(
            2 * context_dim * FUTURE_LENGTH, 2 * context_dim * FUTURE_LENGTH
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=tf_layer
        )
        self.mlp2 = MLP(
            2 * context_dim * FUTURE_LENGTH, 2 * context_dim * FUTURE_LENGTH
        )
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(
            context_dim, context_dim // 2, context_dim + 3
        )
        self.linear = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)

    def forward(self, x_context, beta):
        x, context = x_context
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)
        pre_trans = final_emb.permute(1, 0, 2)
        pre_trans = pre_trans.reshape(
            final_emb.shape[1], final_emb.shape[0] * final_emb.shape[2]
        )
        pre_trans = self.mlp1(pre_trans)
        pre_trans = pre_trans.reshape(
            final_emb.shape[1], 1, final_emb.shape[0] * final_emb.shape[2]
        )
        trans = self.transformer_encoder(pre_trans)
        trans = trans.reshape(
            final_emb.shape[1], final_emb.shape[0] * final_emb.shape[2]
        )
        trans = self.mlp2(trans)
        trans = trans.reshape(
            final_emb.shape[1], final_emb.shape[0], final_emb.shape[2]
        )
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class TransformerLinear(Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim + 3, 128)
        self.layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=2, dim_feedforward=512
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1, 0, 2)
        # pdb.set_trace()
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(
            1, 0, 2
        )  # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)


class LinearDecoder(Module):
    def __init__(self):
        super().__init__()
        self.act = F.leaky_relu
        self.layers = ModuleList(
            [
                # nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                # nn.Linear(2, 64),
                # nn.Linear(2, 64),
            ]
        )

    def forward(self, code):
        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out


class DiffusionTraj(Module):
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, batch_size=None, attn_mask=None, loss_mask=None, t=None):
        total_agent_num, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(total_agent_num)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()  # (B, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)

        e_theta = self.net([c0 * x_0 + c1 * e_rand, context], beta=beta, batch_size=batch_size, mask=attn_mask)
        # loss = F.mse_loss(
        #     e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction="mean"
        # )
        all_loss = F.mse_loss(
            e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction="none"
        )
        if batch_size is not None:
            all_loss = all_loss.reshape((batch_size, -1, 2))
        if loss_mask is not None:
            reshaped_mask = ~loss_mask.unsqueeze(-1).expand(-1, -1, 2)
            loss = all_loss.masked_select(reshaped_mask)
            loss = torch.mean(loss)
        else:
            loss = torch.mean(all_loss)
        return loss

    def sample_sicnav_inference(
        self,
        num_points,
        context,
        sample,
        bestof,
        point_dim=2,
        flexibility=0.0,
        ret_traj=False,
        sampling="ddpm",
        step=100,
        with_constraints=True,
        dynamics=None,
    ):
        """
        context: B x 256
        """
        traj_list = []
        sample_context = context.repeat(sample, 1)  # B*sample x 256
        batch_size = sample_context.size(0)
        if bestof:
            x_T = torch.randn([batch_size, num_points, point_dim]).to(
                sample_context.device
            )
        else:
            x_T = torch.zeros([batch_size, num_points, point_dim]).to(
                sample_context.device
            )
        traj = {self.var_sched.num_steps: x_T}
        stride = int(100 / step)
        for t in range(self.var_sched.num_steps, 0, -stride):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t - stride]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net([x_t, sample_context], beta=beta)
            if sampling == "ddpm":
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            elif sampling == "ddim":
                x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_next = (
                    alpha_bar_next.sqrt() * x0_t
                    + (1 - alpha_bar_next).sqrt() * e_theta
                )
            else:
                pdb.set_trace()
            traj[t - stride] = x_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            traj_list.append(traj)
        else:
            traj_list.append(traj[0])
        number_of_steps = sample * (self.var_sched.num_steps // stride + 1)
        return traj_list[0].reshape(sample, context.size(0), -1, 2), number_of_steps


    def sample(
        self,
        num_points,
        context,
        sample,
        bestof,
        point_dim=2,
        flexibility=0.0,
        ret_traj=False,
        sampling="ddpm",
        step=100,
        with_constraints=True,
        constraint_type=None,
        dynamics=None,
    ):
        traj_list = []
        for i in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, num_points, point_dim]).to(
                    context.device
                )
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(
                    context.device
                )
            traj = {self.var_sched.num_steps: x_T}

            stride = int(100 / step)
            for t in range(self.var_sched.num_steps, 0, -stride):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t - stride]
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t] * batch_size]
                e_theta = self.net([x_t, context], beta=beta)
                if sampling == "ddpm":
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = (
                        alpha_bar_next.sqrt() * x0_t
                        + (1 - alpha_bar_next).sqrt() * e_theta
                    )
                else:
                    pdb.set_trace()
                traj[t - stride] = x_next.detach()  # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
                if not ret_traj:
                    del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
        number_of_steps = sample * (self.var_sched.num_steps // stride + 1)
        # TODO(fumiaki): We should correct some outputs.
        return (
            torch.stack(traj_list),
            number_of_steps,
            0, # num_iter,
            0, # total_num_samples_with_collisions,
            0, # total_num_samples_with_collisions_fixed,
        )


class SmallMLP(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.layer1 = nn.Linear(
            12 * point_dim + context_dim + 1, 512
        )  # I think the three inputs should be split up
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 12 * point_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x_context, beta):
        x, context = x_context
        batch_size = x.size(0)
        x_size = x.size()
        x = x.view(batch_size, -1)  # (B, timesteps x 2)
        context = context.view(batch_size, -1)  # (B, F)
        t = beta.view(batch_size, 1)  # (B, 1)
        x = torch.cat((x, context, t), 1)  # (B, timesteps x 2 + F + 1)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = x.view(x_size)

        return x


class BigMLP(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.layer1 = nn.Linear(
            12 * point_dim + context_dim + 1, 512
        )  # I think the three inputs should be split up
        self.layer2 = nn.Linear(512, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 1024)
        self.layer5 = nn.Linear(1024, 1024)
        self.layer6 = nn.Linear(1024, 1024)
        self.layer7 = nn.Linear(1024, 1024)
        self.layer8 = nn.Linear(1024, 1024)
        self.layer9 = nn.Linear(1024, 1024)
        self.layer10 = nn.Linear(1024, 1024)
        self.layer11 = nn.Linear(1024, 512)
        self.layer12 = nn.Linear(512, 12 * point_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x_context, beta):
        x, context = x_context
        batch_size = x.size(0)
        x_size = x.size()
        x = x.view(batch_size, -1)  # (B, timesteps x 2)
        context = context.view(batch_size, -1)  # (B, F)
        t = beta.view(batch_size, 1)  # (B, 1)
        x = torch.cat((x, context, t), 1)  # (B, timesteps x 2 + F + 1)

        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.activation(self.layer5(x))
        x = self.activation(self.layer6(x))
        x = self.activation(self.layer7(x))
        x = self.activation(self.layer8(x))
        x = self.activation(self.layer9(x))
        x = self.activation(self.layer10(x))
        x = self.activation(self.layer11(x))
        x = self.layer12(x)
        x = x.view(x_size)

        return x
