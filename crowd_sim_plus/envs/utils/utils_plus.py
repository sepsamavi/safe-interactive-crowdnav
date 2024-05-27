import torch
import numpy as np
from copy import deepcopy


def  intersection_of_vec_line_and_2p_line(vorig_x, vorig_y, vx, vy, x1, y1, x2, y2):
    x3 = vorig_x
    y3 = vorig_y
    x4 = vorig_x + vx
    y4 = vorig_y + vy

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / \
         ( (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4) )

    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / \
         ( (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4) )

    return px, py


def closest_point_on_segment(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

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
    return x, y

def closest_point_on_segment_extended(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    # if u > 1:
    #     u = 1
    # elif u < 0:
    #     u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py
    return x, y

def dist_point_to_line_seg(point, line_p1, line_p2):


    return np.inf


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

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


# only for walls right now where x1, y1, x2, y2 are floats
def point_to_segment_dist_vectorized(x1, y1, x2, y2, x3, y3, device):
        """
        Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

        """
        px = x2 - x1
        py = y2 - y1

        if px == 0 and py == 0:
            return torch.norm(torch.Tensor([x3-x1, y3-y1], device=device))

        # zero_components = torch.logical_and(px==0, py==0)
        # non_zero_components = torch.logical_not(zero_components)

        # I'm not checking if it's equal to batch size because that should basically never happen. if it does it's likely an error somewhere else
        # if torch.sum(zero_components) > 0:
        dist_zero_calc = torch.norm(torch.stack([x3 - x1, y3 - y1], dim=1), dim=1)

        u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

        u[u>1] = 1
        u[u<0] = 0

        # if u > 1:
        #     u = 1
        # elif u < 0:
        #     u = 0

        # (x, y) is the closest point to (x3, y3) on the line segment
        x = x1 + u * px
        y = y1 + u * py

        dist_non_zero_calc = torch.norm(torch.stack([x - x3, y-y3], dim=1), dim=1)

        return dist_non_zero_calc

def find_intersection_of_static_obstacles(line1, line2):
    if line1[0][0] == line2[0][0] and line1[0][1] == line2[0][1]:
        return (line1[0][0], line1[0][1])

    elif line1[0][0] == line2[1][0] and line1[0][1] == line2[1][1]:
        return line1[0][0], line1[0][1]

    elif line1[1][0] == line2[1][0] and line1[1][1] == line2[1][1]:
        return line1[1][0], line1[1][1]

    elif line1[1][0] == line2[0][0] and line1[1][1] == line2[0][1]:
        return line1[1][0], line1[1][1]

    else:
        print("Error in finding intersection of static obstacles")
        exit()

    line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    line2[0][0], line2[0][1], line2[1][0], line2[1][1]

def check_point_on_line_seg(x1,y1,x2,y2, px,py, static_obs_int_x, static_obs_int_y):#scaling, dirx, diry):
    """
    function assumses that px and py must exist on the line drawn from x1,y1 to x2,y2

    the scaling param considers whether or not we need to "extend the line"
    """
    x1_og = deepcopy(x1)
    x2_og = deepcopy(x2)

    if static_obs_int_x == x1:
        if x2>x1:
            x2 = np.inf
        else:
            x2 = -np.inf

        if y2>y1:
            y2 = np.inf
        else:
            y2 = -np.inf
    elif static_obs_int_x == x1:
        if x1 > x2:
            x1 = np.inf
        else:
            x1 = -np.inf

        if y1>y2:
            y1 = np.inf
        else:
            y1 = -np.inf

    if x1_og != x2_og:
        min_x = min(x1,x2)
        max_x = max(x1,x2)

        on_line = ((px>=min_x) and (px<= max_x))

        # if scaling > 0:
        #     on_line = on_line or (px > max_x and dirx > 0) or (px < min_x and dirx < 0)

    else:
        min_y = min(y1,y2)
        max_y = max(y1,y2)

        on_line = ((py>=min_y) and (py<= max_y))

        # if scaling > 0:
        #     on_line = on_line or (py > max_y and diry > 0) or (py < min_y and diry < 0)
    return on_line

# James's closest distance code for constraining agent actions against static obstacles
def closest_distance_between_line_segments(a0,a1,b0,b1):
    """Given two lines defined by numpy.array pairs (a0,a1,b0,b1), return the closest points on each line segment and their distance. Adapted from https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments


    :param a0: start of obstacle
    :param a1: end of obstacle
    :param b0: start pos of robot
    :param b1: end pos of robot
    :return: _description_
    """

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    if magA < 1e-8:
        a1 = a0
        A = _A = np.zeros_like(A)
    else:
        _A = A / magA
    if magB < 1e-8:
        b1 = b0
        B = _B = np.zeros_like(B)
    else:
        _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2

    if not denom:
        d0 = np.dot(_A,(b0-a0))

        # Overlap only possible with clamping
        d1 = np.dot(_A,(b1-a0))

        if d0 <= 0 >= d1:
            # Is segment B before A?
            if np.absolute(d0) < np.absolute(d1):
                return a0,b0,np.linalg.norm(a0-b0)
            return a0,b1,np.linalg.norm(a0-b1)
        elif d0 >= magA <= d1:
            # Is segment B after A?
            if np.absolute(d0) < np.absolute(d1):
                return a1,b0,np.linalg.norm(a1-b0)
            return a1,b1,np.linalg.norm(a1-b1)
        else:
            # There is an overlap of the parallel segments, there are are multiple (uncountably infinite) closest points.
            # We use the conventions below to select pA and pB

            # first make sure that the parallel vectors are in the same direction
            if np.linalg.norm(_A-_B) < 1e-8 or magB < 1e-8:
                a0f = a0
                a1f = a1
                Af = A
                _Af = _A
            else:
                a0f = a1
                a1f = a0
                Af = -A
                _Af = -_A

            d0f = np.dot(_Af,(b0-a0f))
            d1f = np.dot(_Af,(b1-a0f))
            # assert d1f >= 0, "Should only happen when d1f >=0 but d1f = %f" % d1f
            if d0f >=0:
                # B is fully covered by A 0 <= d0f <= d1f
                # OR
                # b0 is in between a0f and a1f but b1 is to the right (outside) of a1f 0 >= d0 <= d1 >= magA
                pB = b0
                W = pB - a0f
                t = np.dot(_Af,W)
                assert t <= magA and t >= 0.0, "Should have 0 <= t <= 1 but t = %f" % t
                pA = a0f + _Af*t
            elif d0f < 0:
                # B fully covers A d0f < 0 and d1f >= magA
                # OR
                # b0 is to the left (outside) of a0f but b1 is in between a0f and a1f d0f < 0 <= d1f <= magA
                pA = a0f
                W = pA - b0
                t = np.dot(_B,W)
                assert t <= magB and t >= 0.0, "Should have 0 <= t <= 1 but t = %f" % t
                pB = b0 + _B*t
            else:
                raise Exception("This should not happen ever here")
            return pA,pB,np.linalg.norm(pA-pB)

        # # Segments overlap, return distance between parallel segments
        # return None,None,np.linalg.norm(((d0*_A)+a0)-b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if t0 < 0:
        pA = a0
    elif t0 > magA:
        pA = a1

    if t1 < 0:
        pB = b0
    elif t1 > magB:
        pB = b1

    # Clamp projection A
    if (t0 < 0) or (t0 > magA):
        dot = np.dot(_B,(pA-b0))
        if dot < 0:
            dot = 0
        elif dot > magB:
            dot = magB
        pB = b0 + (_B * dot)

    # Clamp projection B
    if (t1 < 0) or (t1 > magB):
        dot = np.dot(_A,(pB-a0))
        if dot < 0:
            dot = 0
        elif dot > magA:
            dot = magA
        pA = a0 + (_A * dot)


    return pA,pB,np.linalg.norm(pA-pB)