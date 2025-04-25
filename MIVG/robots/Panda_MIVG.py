import numpy as np
import swift  # Import Swift simulation environment for robot simulation
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3
from typing import Union, List
from roboticstoolbox.robot.Link import Link
from numpy.typing import ArrayLike
import spatialgeometry as sg
import spatialmath as sm

class Panda_guide(Robot):
    def __init__(self):
        # Load Panda robot model from URDF file
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "franka_description/robots/panda_arm_hand.urdf.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Franka Emika",
            gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # Set the gripper's tool transformation
        self.grippers[0].tool = SE3(0, 0, 0.1034)

        # Define joint velocity limits
        self.qdlim = np.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        )

        # Define configurations
        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def link_collision_damper(
        self,
        shape,
        q: ArrayLike,
        di: float = 0.3,
        ds: float = 0.03,
        xi: float = 1.0,
        r_guide: float = 0.08,  # New parameter: guiding radius
        end: Union["Link", None] = None,
        start: Union["Link", None] = None,
        collision_list: Union[List, None] = None,
    ):
        # Get start and end links for collision damping
        end, start, _ = self._get_limit_links(start=start, end=end)
        links, n, _ = self.get_path(start=start, end=end)

        q = np.array(q)
        j = 0
        Ain = None
        bin = None
        min_distance = float('inf')

        # Initialize variables for Aje and bv
        Aje = None
        bv = None

        # Initialize status variable
        status = "Normal"

        def pad_to_7_cols(M):
            """
            Extend the matrix M to 7 columns, padding with zeros if necessary.
            Raise an error if M has more columns than expected.
            """
            if M.shape[1] < n:
                diff = n - M.shape[1]
                M = np.pad(M, ((0, 0), (0, diff)), mode='constant', constant_values=0.0)
            elif M.shape[1] > n:
                raise ValueError(
                    f"Matrix has more columns ({M.shape[1]}) than 9 which shouldn't happen for a 9-DoF arm."
                )
            return M

        def indiv_calculation(link, link_col, q):
            """
            Perform individual calculation for each link, including computing Ain, bin, Aje, bv, and minimum distance.
            """
            d, wTlp, wTcp = link_col.closest_point(shape, di)
            if d is None or wTlp is None or wTcp is None:
                return None, None, float('inf'), None, None

            lpTcp = -wTlp + wTcp
            norm = lpTcp / d if d > 0 else np.zeros(3)
            norm_h = np.expand_dims(np.concatenate((norm, [0.0, 0.0, 0.0])), axis=0)

            Je = self.jacobe(q, start=start, end=link, tool=link_col.T)
            n_dim = Je.shape[1]

            dp = norm_h @ shape.v
            l_Ain = np.zeros((1, n))
            l_Ain[0, :n_dim] = norm_h @ Je
            l_bin = (xi * (d - ds) / (di - ds)) + dp

            if d <= r_guide:
                # Calculate Aje and bv when within guiding radius
                direction_vector = lpTcp / np.linalg.norm(lpTcp) if np.linalg.norm(lpTcp) > 1e-12 else np.array([0, 0, 1])
                obs_dir = shape.v[:3] / np.linalg.norm(shape.v[:3]) if np.linalg.norm(shape.v[:3]) > 1e-12 else np.array([1, 0, 0])
                perp_vec = np.cross(direction_vector, obs_dir)
                if np.linalg.norm(perp_vec) < 1e-12:
                    perp_vec = np.cross(direction_vector, np.array([0, 0, 1]))
                perp_vec = perp_vec / np.linalg.norm(perp_vec)

                l_bv = np.concatenate((perp_vec, [0.0, 0.0, 0.0]))
                Je = pad_to_7_cols(Je)
                l_Aje = Je
                l_bv = l_bv.reshape(-1, 1)  # (6, 1)
            else:
                # When d > r_guide, skip Aje and bv calculation
                l_Aje = None
                l_bv = None

            return l_Ain, l_bin, d, l_Aje, l_bv

        for link in links:
            if link.isjoint:
                j += 1

            if collision_list is None:
                col_list = link.collision
            else:
                col_list = [collision_list[j - 1]]

            for link_col in col_list:
                l_Ain, l_bin, d, l_Aje, l_bv = indiv_calculation(link, link_col, q)
                if d is not None:
                    min_distance = min(min_distance, d)
                    if d <= r_guide:
                        # Update status to "Bypassing"
                        status = "Bypassing"

                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = np.concatenate((Ain, l_Ain))

                    if bin is None:
                        bin = np.array(l_bin)
                    else:
                        bin = np.concatenate((bin, l_bin))

                if l_Aje is not None and l_bv is not None:
                    if Aje is None:
                        Aje = l_Aje
                        bv = l_bv
                    else:
                        Aje = np.concatenate((Aje, l_Aje), axis=0)
                        bv = np.concatenate((bv, l_bv), axis=0)

        # Return with updated status
        return Ain, bin, min_distance, Aje, bv, status


if __name__ == "__main__":
    panda = Panda_guide()
    # Set the initial joint configuration
    panda.q = panda.qr

    # Define obstacle and its velocity
    s0 = sg.Sphere(radius=0.05, pose=sm.SE3(0.26, 0.2, 0.60))
    s0.v = np.array([0, -0.2, 0, 0, 0, 0])  # First 3 components are linear velocity

    env = swift.Swift()
    env.launch()
    env.add(panda)
    env.add(s0)

    Ain, bin, min_dist, Aje, bv, status = panda.link_collision_damper(
        s0,
        q=panda.q,
        di=0.3,
        ds=0.03,
        xi=1.0,
        start=panda.link_dict["panda_link1"],
        end=panda.link_dict["panda_hand"]
    )

    print("Ain shape:", Ain.shape if Ain is not None else None)
    print("bin shape:", bin.shape if bin is not None else None)
    print("Aje shape:", Aje.shape if Aje is not None else None)
    print("bv shape:", bv.shape if bv is not None else None)
    print("min_distance:", min_dist)
    print("status:", status)
