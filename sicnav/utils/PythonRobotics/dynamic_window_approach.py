"""

Python Implementation of the Dynamic Window Approach (DWA)
D. Fox, W. Burgard, and S. Thrun,
“The dynamic window approach to collision avoidance,” IEEE RAM, Vol. 4, no. 1, 1997.

The class and method structure in this file is adapted from the PythonRobotics DWA implementaiton
by Atsushi Sakai (@Atsushi_twi) and Göktuğ Karakaşlı. However, the algorithm has been completely
re-written in the calc_dynamic_window and calc_control_and_trajectory functions to match the
original DWA publication cited above.

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import logging

show_animation = True


def dwa_control(x, config, goal, ob, line_obs=[]):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u = calc_control_and_trajectory(x, dw, config, goal, ob, line_obs)

    return u


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check
        self.robot_width = 0.4  # [m] for collision check
        self.robot_length = 0.6  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = []
        self.line_obs = []

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """
    v = u[0]
    w = u[1]
    if abs(w) < 0.01:
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
    else:
        theta = x[2]
        theta2 = theta + w * dt
        x[0] += (v / w) * (np.sin(theta2) - np.sin(theta))
        x[1] += (v / w) * (np.cos(theta) - np.cos(theta2))
        x[2] = theta2
    x[3] = u[0]
    x[4] = u[1]
    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    logging.debug('V: {} W: {}'.format(x[3], x[4]))
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    v_max = min(Vs[1], Vd[1])
    v_min = max(Vs[0], Vd[0])
    if v_min > v_max - config.max_accel * config.dt:
        v_min = v_max - config.max_accel * config.dt

    yaw_rate_max = min(Vs[3], Vd[3])
    yaw_rate_min = max(Vs[2], Vd[2])
    if yaw_rate_min > yaw_rate_max - config.max_delta_yaw_rate * config.dt:
        yaw_rate_min = yaw_rate_max - config.max_delta_yaw_rate * config.dt
    dw = [v_min, v_max, yaw_rate_min, yaw_rate_max]

    logging.debug('DW: {}'.format(dw))

    return dw


def predict_trajectory(x_init, v, y, goal, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time and np.linalg.norm(x[0:2] - goal) > config.robot_radius * 0.5:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob, line_obs=[]):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]

    # evaluate all trajectory with sampled input in dynamic window
    # traj_data = []
    # fig = plt.figure()
    # ax = fig.add_subplot()

    # **Note: we're doing a maximization now like in the DWA paper.
    speeds = np.arange(dw[0], dw[1]+1e-3, config.v_resolution)
    yaw_rates = np.arange(dw[2], dw[3]+1e-3, config.yaw_rate_resolution)

    H = np.zeros((speeds.shape[0], yaw_rates.shape[0]))  # head
    D = np.zeros((speeds.shape[0], yaw_rates.shape[0]))  # dist
    V = np.zeros((speeds.shape[0], yaw_rates.shape[0]))  # vel
    admissible = np.ones((speeds.shape[0], yaw_rates.shape[0]))

    logging.debug('yaw rates: '+str(yaw_rates))

    for vidx, v in enumerate(speeds):
        for yidx, y in enumerate(yaw_rates):

            trajectory = predict_trajectory(x_init, v, y, goal, config)
            # Check if (v,w) is admissible:
            dist = calc_obstacle_cost(trajectory, ob, config, line_obs) # distance to closest object that intersects with trajectory
            # check if there is enough time to stop before colliding with the obstacle
            if v > np.sqrt(2 * dist * config.max_accel) or y > np.sqrt(2 * dist * config.max_delta_yaw_rate):
                admissible[vidx, yidx] = 0
                continue
            D[vidx, yidx] = dist
            H[vidx, yidx] = calc_to_goal_cost(trajectory, goal)
            V[vidx, yidx] = v
    hmax = np.pi
    dmax = config.max_d
    vmax = config.max_speed
    H /= hmax
    D /= dmax
    V /= vmax
    T = config.to_goal_cost_gain * H + config.obstacle_cost_gain * D + config.speed_cost_gain * V
    vidx, yidx = np.unravel_index(np.argmax(T), T.shape)
    if not admissible[vidx, yidx]:
        logging.debug('NONE ADMISSIBLE')
        return [0, 0]
    else:
        logging.debug('H: {} D: {} V: {}'.format(H[vidx, yidx], D[vidx, yidx], V[vidx, yidx]))
        u = [speeds[vidx], yaw_rates[yidx]]
        if abs(u[0]) < config.robot_stuck_flag_cons and abs(x[3]) < config.robot_stuck_flag_cons:
            u[1] = -config.max_delta_yaw_rate
        return u


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    Adapted from CrowdNav Project: https://github.com/vita-epfl/CrowdNav
    """

    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


def calc_obstacle_cost(trajectory, ob, config, line_obs=[]):
    dist = config.max_d
    x_init = trajectory[0]
    for i in range(trajectory.shape[0]): # for each step in the arc's trajectory
        x = trajectory[i, 0]
        y = trajectory[i, 1]
        for j in range(ob.shape[0]): # for each obstacle
            ox = ob[j, 0]
            oy = ob[j, 1]
            orad = ob[j, 2]
            # if the obstacle intersects with a point on the arc. i.e. there is a collision on the arc
            if np.sqrt((x - ox)**2 + (y - oy)**2) < config.robot_radius + orad + 0.01:
                # calculate distance to obstacle from robot's current position
                d_obs = np.sqrt((x_init[0] - ox)**2 + (x_init[1] - oy)**2)
                # if this distance is lower than the minimum distance, the minimum distance is this.
                if d_obs < dist:
                    dist = d_obs

        for j, line in enumerate(line_obs): # for each line obstacle obstacle
            arc_delta_obj = point_to_segment_dist(line[0][0], line[0][1], line[1][0], line[1][1], x, y)
            # if the obstacle intersects with a point on the arc. i.e. there is a collision on the arc
            if arc_delta_obj < config.robot_radius + 0.02:
                # calculate distance to obstacle from robot's current position
                d_obs = point_to_segment_dist(line[0][0], line[0][1], line[1][0], line[1][1], x_init[0], x_init[1])
                # if this distance is lower than the minimum distance, the minimum distance is this.
                if d_obs < dist:
                    dist = d_obs
    return dist


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost_angle = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    head = np.pi - cost_angle
    return head


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)
