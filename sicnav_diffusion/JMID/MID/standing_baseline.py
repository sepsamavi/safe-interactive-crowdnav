import numpy as np

from baseline import Baseline


class StandingBaseline(Baseline):
    def _generate(self, history, **kwargs):
        last_x_y = np.expand_dims(history[:, -1, :2], axis=1)  # B x 1 x 2
        predictions = np.tile(last_x_y, (1, kwargs["num_points"], 1))
        return np.expand_dims(predictions, axis=0)  # 1 x B x 12 x 2
