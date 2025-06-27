import dill
import numpy as np

from mid import MID
from utils.trajectron_hypers import get_traj_hypers


class CVMM:
    def __init__(self, env):
        self._dt = env.scenes[0].dt  # seconds per frame

    def eval(self):
        """
        Placeholder method. This is for setting PyTorch models to eval mode.
        """
        pass

    def generate(self, batch, *args, **kwargs):
        """
        Returns predictions and "number of steps" required to compute the predictions.
        Since this is not an iterative method, 0 is returned as the number of steps.
        """
        x_t = batch[1]  # B x hist x state
        last_x_y = np.expand_dims(x_t[:, -1, :2], axis=1)  # B x 1 x 2
        last_x_y_vel = np.expand_dims(x_t[:, -1, 2:4], axis=1)  # B x 1 x 2
        last_x_y_vel_tiled = np.tile(last_x_y_vel, (1, kwargs["num_points"], 1))
        time_elapsed = np.arange(1, kwargs["num_points"] + 1) * self._dt
        distance_traveled = (
            np.expand_dims(np.expand_dims(time_elapsed, axis=0), axis=2)
            * last_x_y_vel_tiled
        )
        predictions = distance_traveled + last_x_y
        return np.expand_dims(predictions, axis=0), 0  # 1 x B x pred_horiz x 2


class ConstantVelocityBaseline(MID):
    def _build(self):
        self._build_dir()
        self._build_hyperparams()
        self._build_model()
        self._build_eval_loader()
        self._build_inference_scenes()
        print("> Everything built. Have fun :)")

    def _build_hyperparams(self):
        self.hyperparams = get_traj_hypers()
        self.hyperparams["enc_rnn_dim_edge"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_edge_influence"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_history"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_future"] = self.config.encoder_dim // 2

        # Use below settings for conditioning MID on robot future motion
        # self.hyperparams["enc_rnn_dim_edge"] = self.config.encoder_dim // 4
        # self.hyperparams["enc_rnn_dim_edge_influence"] = self.config.encoder_dim // 4
        # self.hyperparams["enc_rnn_dim_history"] = self.config.encoder_dim // 4
        # self.hyperparams["enc_rnn_dim_future"] = self.config.encoder_dim // 8

    def _build_model(self):
        with open(self.eval_data_path, "rb") as f:
            self.eval_env = dill.load(f, encoding="latin1")
        self.model = CVMM(self.eval_env)
        print("> Model built!")
