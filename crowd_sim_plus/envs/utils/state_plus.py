class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, omega=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta
        self.omega = omega

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        if self.omega is None:
            return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.omega)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta, self.omega]])

# Overwriting to account for overwritten FullState
class JointState(object):
    def __init__(self, self_state, human_states, static_obs=[]):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
        self.static_obs = static_obs


class FullyObservableJointState(object):
    def __init__(self, self_state, human_states, static_obs=[]):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, FullState)

        self.self_state = self_state
        self.human_states = human_states
        self.static_obs = static_obs


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

