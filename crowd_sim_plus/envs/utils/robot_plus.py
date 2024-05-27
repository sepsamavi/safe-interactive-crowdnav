import logging
import numpy as np
from crowd_sim_plus.envs.utils.agent_plus import Agent
from crowd_sim_plus.envs.utils.state_plus import JointState, FullyObservableJointState, FullState, ObservableState

def rot_2D(theta_vi, p_ai_i):
    Rot_vi = np.array([[np.cos(theta_vi), np.sin(theta_vi)],
                      [-np.sin(theta_vi), np.cos(theta_vi)]])
    p_av_v = np.dot(Rot_vi, p_ai_i)
    return p_av_v

def tsf_2D(q_vi_i, p_ai_i):
    theta_vi = q_vi_i[2]
    return rot_2D(theta_vi, p_ai_i - q_vi_i[0:2])

def general_tsf_2D(R, t, p):
    return np.matmul(R, p) + t

def rot_tsf_2D(R, p):
    return np.matmul(R, p)

def get_vizable_robocentric(robocentric_states):
    new_bulk_states = []
    rot_ang = -np.pi / 2
    for bulk_state in robocentric_states:
        robot_state = bulk_state[0]
        new_robot_pos = rot_2D(rot_ang, np.array(robot_state.position))
        new_robot_gpos = rot_2D(rot_ang, np.array(robot_state.goal_position))
        new_robot_vel = rot_2D(rot_ang, np.array(robot_state.velocity))
        new_robot_state = FullState(px=new_robot_pos[0], py=new_robot_pos[1],
                                    vx=new_robot_vel[0], vy=new_robot_vel[1],
                                    radius=robot_state.radius,
                                    gx=new_robot_gpos[0], gy=[new_robot_gpos[1]],
                                    v_pref=robot_state.v_pref,
                                    theta=-rot_ang, omega=robot_state.omega)

        human_states = bulk_state[1]
        new_human_states = []
        for human_state in human_states:
            new_human_pos = rot_2D(rot_ang, np.array(human_state.position))
            new_human_vel = rot_2D(rot_ang, np.array(human_state.velocity))
            new_human_states.append(ObservableState(px=new_human_pos[0], py=new_human_pos[1],
                                                    vx=new_human_vel[0], vy=new_human_vel[1],
                                                    radius=human_state.radius))

        static_obs = bulk_state[2]
        new_static_obs = []
        for static_ob in static_obs:
            start_pos = tuple(rot_2D(rot_ang, np.array(static_ob[0])).tolist())
            final_pos = tuple(rot_2D(rot_ang, np.array(static_ob[1])).tolist())
            new_static_obs.append([start_pos, final_pos])

        new_bulk_states.append([new_robot_state, new_human_states, new_static_obs])
    return new_bulk_states


# Overwriting Robot.
class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob, static_obs=[]):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        # state = JointState(self.get_full_state(), ob, static_obs)
        state = self.get_joint_state(ob, static_obs)
        action = self.policy.predict(state)
        return action

    def get_joint_state(self, ob, static_obs=[]):
        return JointState(self.get_full_state(), ob, static_obs)

    def get_robocentric_state(self, ob, static_obs=[]):
        robot_state = self.get_full_state()
        theta_ri = robot_state.theta
        q_ri_i = np.array([robot_state.px, robot_state.py, theta_ri])
        if abs(np.arctan2(robot_state.vy, robot_state.vx) - theta_ri) % 2*np.pi > 1e-15 and abs(np.arctan2(robot_state.vy, robot_state.vx) - theta_ri) % np.pi > 1e-15 and np.linalg.norm(robot_state.velocity) - 0 > 1e-15:
            logging.warn('Vel angle vs theta_ri inconsistency. In deg: theta_ri = {:}, atan2(.,.) = {:}'.format(str(theta_ri*180/np.pi), str(np.arctan2(robot_state.vy, robot_state.vx)*180/np.pi)))
        # Calculate new robot state
        new_robot_pos = tsf_2D(q_ri_i, np.array(robot_state.position))
        new_robot_gpos = tsf_2D(q_ri_i, np.array(robot_state.goal_position))
        new_robot_vel = rot_2D(theta_ri, np.array(robot_state.velocity))
        if abs(new_robot_vel[1] - 0) > 1e-15:
            logging.warn('There is a problem with the rotation calculation')
        new_robot_state = FullState(px=new_robot_pos[0], py=new_robot_pos[1],
                                    vx=new_robot_vel[0], vy=new_robot_vel[1],
                                    radius=robot_state.radius,
                                    gx=new_robot_gpos[0], gy=new_robot_gpos[1],
                                    v_pref=robot_state.v_pref,
                                    theta=0.0, omega=robot_state.omega)

        new_ob = []
        for human_state in ob:
            new_human_pos = tsf_2D(q_ri_i, np.array(human_state.position))
            new_human_vel = rot_2D(theta_ri, np.array(human_state.velocity))

            new_ob.append(ObservableState(px=new_human_pos[0], py=new_human_pos[1],
                                          vx=new_human_vel[0], vy=new_human_vel[1],
                                          radius=human_state.radius))

        new_static_obs = []
        for static_ob in static_obs:
            start_pos = tuple(tsf_2D(q_ri_i, np.array(static_ob[0])).tolist())
            final_pos = tuple(tsf_2D(q_ri_i, np.array(static_ob[1])).tolist())
            new_static_obs.append([start_pos, final_pos])

        return JointState(new_robot_state, new_ob, new_static_obs)

    def get_robocentric_state_x_goal_aligned(self, ob, static_obs=[]):
        """
        Transform world frame into a robocentric frame. This new frame has the positive x-axis oriented to robot's goal.
        The robot's new position will always be (0,0) in this new frame
        Inputs:
        -----------
        ob - dictionary of humans in the scence  (humans are full_state objects)
        static_obs not handled yet

        Returns:
        --------
        new_robot_state - numpy array in format of SB3 observation return
        new_ob - list of numpy arrays for human in partially observable state

        """
        robot_state = self.get_full_state(original=True)

        # vector transformation from world frame to robot frame which produces R_w_r from is coordinate transform of robot to world (not what we want but need to calc first)
        tx = robot_state.px
        ty = robot_state.py
        goal_state = robot_state.goal_position
        rot_amount = np.arctan2(goal_state[1] - ty, goal_state[0]-tx)
        R_w_r = np.array([[np.cos(rot_amount), -np.sin(rot_amount)],
                          [np.sin(rot_amount), np.cos(rot_amount)]])
        O_w_r = np.array([tx,ty])

        # calculuate corresponding R_r_w which is a coordinate transform from world to robot
        R_r_w = np.transpose(R_w_r)
        O_r_w = np.matmul(R_r_w, -O_w_r)

        # Calculate new robot state
        new_robot_pos = general_tsf_2D(R_r_w, O_r_w, robot_state.position)
        new_robot_gpos = general_tsf_2D(R_r_w, O_r_w, np.array(goal_state))
        new_robot_vel = rot_tsf_2D(R_r_w,robot_state.velocity)
        new_theta = robot_state.theta - rot_amount

        # Construct robot state
        new_robot_state = np.array([new_robot_pos[0], new_robot_pos[1], new_robot_vel[0], new_robot_vel[1], robot_state.radius, new_robot_gpos[0], new_robot_gpos[1], robot_state.v_pref, new_theta], dtype=np.float32)

        # basic checks
        if abs(new_robot_gpos[0] - np.linalg.norm(np.array(robot_state.position) - np.array(goal_state)) ) > 1e-10:
            logging.warn('There is a problem with the rotation calculation')
            # exit()
        if abs(new_robot_gpos[1]) > 1e-5:
            logging.warn('There is a problem with the rotation calculation, here')
            # exit()

        # Construct human states and ob
        new_ob = []
        for human_state in ob:
            new_human_pos = general_tsf_2D(R_r_w, O_r_w, np.array(human_state.position))
            new_human_vel = rot_tsf_2D(R_r_w, np.array(human_state.velocity))
            new_ob.append(np.array([new_human_pos[0], new_human_pos[1], new_human_vel[0], new_human_vel[0], human_state.radius], dtype=np.float32))


        return new_robot_state, new_ob

    def map_occlusion(self, ob):
        """
        Apply occlusions to state space for robot.
        This can only be run after robotcentric transformation
        Inputs:
        -----------
        ob - dictionary of humans in the scence  (humans are of type np.array)

        Returns:
        --------
        ob_occluded - dictionary of non-ocluded humans
        num_not_occluded - # of non-occluded humans
        """

        # Sort humans from closest to furthest
        dists = np.array([np.linalg.norm(ob[i][self.position_indexes]) for i in range(len(ob))])
        sorted_humans = np.argsort(dists)

        occluded_regions = []
        non_occluded_humans = []
        for idx, sorted_human in enumerate(sorted_humans):

            L, R, duplication = self.get_human_FOV_blocked(ob[sorted_human])

            if idx == 0:
                occluded_regions.append([L,R])
                if len(duplication) != 0:
                    occluded_regions.append(duplication)
                non_occluded_humans.append(sorted_human)
                continue

            L_bound = L
            R_bound = R
            original_view = R-L
            occluded = False

            visible_threshold = 0.75
            for region in occluded_regions:
                # fully occluded case
                if L_bound >= region[0] and R_bound <= region[1]:
                    occluded = True
                    break
                elif L_bound >= region[0] and L_bound < region[1]:
                    L_bound = region[1]
                elif R_bound > region[0] and R_bound <= region[1]:
                    R_bound = region[0]

                bounded_region = R_bound - L_bound
                if bounded_region/original_view < visible_threshold:
                    occluded=True
                    break

            if occluded is False and len(duplication) > 0:
                L_bound = duplication[0]
                R_bound = duplication[1]
                original_view = R_bound-L_bound

                for region in occluded_regions:
                    # fully occluded case
                    if L_bound >= region[0] and R_bound <= region[1]:
                        occluded = True
                        break
                    elif L_bound >= region[0] and L_bound < region[1]:
                        L_bound = region[1]
                    elif R_bound > region[0] and R_bound <= region[1]:
                        R_bound = region[0]

                    bounded_region = R_bound - L_bound
                    if bounded_region/original_view < visible_threshold:
                        occluded=True
                        break


            if not occluded:
                non_occluded_humans.append(sorted_human)

            occluded_regions.append([L,R])
            if len(duplication) > 0:
                occluded_regions.append(duplication)


        num_not_occluded = len(non_occluded_humans)
        ob_return = []

        for i in range(num_not_occluded):
            ob_return.append(ob[non_occluded_humans[i]])

        return ob_return, num_not_occluded, non_occluded_humans

    def get_human_FOV_blocked(self, human):
        """
        Calculate L and R theta bounds the human takes up in robot's FOV
        L < R and we go by CCW convention
        Inputs:
        -----------
        human - human info in array form (SB3 form)

        Returns:
        --------
        L,R - Left and Right theta bounds
        """

        # relevant human info
        px, py = human[self.position_indexes]
        r = human[self.radius_index]

        # edge case
        if py == 0:
            if px > 0:
                L = np.arctan2(-r/2,px)
                R = np.arctan2(r/2,px)
            else:
                L = np.arctan2(r/2,px)
                R = 2*np.pi + np.arctan(-r/2, px)
            return L, R, []


        # find line tangent to straight line from origin to robot
        # this line also passes through the human
        m = -px/py
        b = py - m*px

        # find points distance radius away on both sides of px
        x_dist_to_move = np.sqrt(r**2/(m**2+1))

        # change to vector implementation where the slope is [1, m] then change magnitude to R

        xL = px + x_dist_to_move
        xR = px - x_dist_to_move

        yL = m*xL + b
        yR = m*xR + b

        # using new points, calculate bounds

        val1 = np.arctan2(yL, xL)
        val2 = np.arctan2(yR, xR)
        duplication = []
        if np.arctan2(py,px) >= -np.pi and np.arctan2(py,px) <= 0:
            L= val2
            R = val1

            if L > 0:
                R += 2*np.pi

        else:
            L = val1
            R = val2

            if R < 0:
                R += 2*np.pi

        # if we have L within quadrant 3, we need to duplicate it
        if L >= -np.pi and L<= -np.pi/2:
            L_temp = L + 2*np.pi
            if R < 0:
                R_temp = R+2*np.pi
            else:
                R_temp = R
            duplication = [L_temp, R_temp]

        return L, R, duplication




class RobotFullKnowledge(Robot):
    def __init__(self, config, section):
        super().__init__(config, section)

    # Overwrite to return full state for the human
    def get_observable_state(self, original=True):
        return self.get_full_state(original=original)

    def get_joint_state(self, ob, static_obs=[]):
        return FullyObservableJointState(self.get_full_state(), ob, static_obs)

    def act(self, ob, static_obs=[]):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = FullyObservableJointState(self.get_full_state(), ob, static_obs)
        action = self.policy.predict(state)
        return action