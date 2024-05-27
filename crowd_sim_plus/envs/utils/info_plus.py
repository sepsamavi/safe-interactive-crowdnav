class ReachGoal(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Reaching goal'

class Timeout(object):
    def __init__(self, val):
        self.val=val

    def __str__(self):
        return 'Timeout'

class Danger(object):
    def __init__(self, val, min_dist=0):
        self.min_dist = min_dist
        self.val = val

    def __str__(self):
        return 'Too close'

class Frozen(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Frozen'

class Collision(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Collision with another agent'


class WallCollision(object):
    def __init__(self, val):
        self.val=val

    def __str__(self):
        return 'Collision with static obstacle'

class Progress(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Progress towards goal'
    
class AngularSmoothness(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Angular smoothness'

class LinearSmoothness(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Linear smoothness'

class TotalReward(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Reward for this step'

class Done(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'Reward for this step'

class Nothing(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return ''