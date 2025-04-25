import swift  
import spatialgeometry as sg  
import roboticstoolbox as rtb  
import spatialmath as sm  
import numpy as np  
import qpsolvers as qp  
from Panda_MIVG import Panda_guide  

# Define velocity range for obstacles
v_max = -0.2  
v_min = -0.25 

# Initialize the Swift simulation environment
env = swift.Swift()
env.launch()

# Create Panda robot instance
panda = Panda_guide()
panda.q = panda.qr  # Set initial joint configuration

n = 7  # Number of robot joints

# Define obstacles with positions and velocities
s0 = sg.Sphere(radius=0.05, pose=sm.SE3(0.52, 0.4, 0.3))
s0.v = [0, -0.2, 0, 0, 0, 0]

s1 = sg.Sphere(radius=0.05, pose=sm.SE3(0.1, 0.35, 0.65))
s1.v = [0.1, -0.3, 0, 0, 0, 0]

s2 = sg.Sphere(radius=0.05, pose=sm.SE3(0.34, 0.3, 0.42))
s2.v = [0, -0.15, 0.1, 0, 0, 0]

s3 = sg.Sphere(radius=0.05, pose=sm.SE3(0.26, 0.45, 0.52))
s3.v = [0, -0.22, 0, 0, 0, 0]

collisions = [s0, s1, s2, s3]  # List of obstacles

# Define the target position
target = sg.Sphere(radius=0.02, pose=sm.SE3(0.6, -0.2, 0.0))

# Add robot, obstacles, and target to the simulation environment
env.add(panda)
env.add(s0)
env.add(s1)
env.add(s2)
env.add(s3)
env.add(target)

# Set target position for the robot end-effector
Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.T[:3, -1]


def step_MIVG():
    """Compute control step for the robot."""
    global env, Tep, collisions, v_max, v_min

    # Parameter initialization (aligned with the paper)
    λ_init = 3
    λ_adj = 1.2
    α = 0.5
    β = 0.5
    r_guide = 0.08
    d_i = 0.3
    d_s = 0.03

    Te = panda.fkine(panda.q)  # Current end-effector pose
    eTep = Te.inv() * Tep  # Pose error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Desired velocity and arrival check
    v_desired, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)

    # Initialize QP parameters
    Q = np.eye(n + 6)
    Q[:n, :n] *= 0.01
    Q[n:, n:] *= 1 / e
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # Constraints initialization
    Aeq, beq = None, None
    A_in = np.zeros((n + 6, n + 6))
    b_in = np.zeros(n + 6)
    A_in[:n, :n], b_in[:n] = panda.joint_velocity_damper(0.05, 0.9, n)

    J_Local, v_guide = None, None
    mode = "Normal"

    # Iterate through obstacles
    for collision in collisions:
        c_Ain, c_bin, d_min, c_J_Local_matrix, c_v_circum, status = panda.link_collision_damper(
            collision,
            q=panda.q[:n],
            di=d_i,
            ds=d_s,
            xi=1.0,
            r_guide=r_guide,
            start=panda.link_dict["panda_link1"],
            end=panda.link_dict["panda_hand"],
        )

        # Update inequality constraints
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain[:, :n], np.zeros((c_Ain.shape[0], 6))]
            A_in = np.r_[A_in, c_Ain]
            b_in = np.r_[b_in, c_bin]

        # Check for bypassing mode
        if status == "Bypassing":
            mode = "Bypassing"

        # Calculate guiding velocity
        if status == "Bypassing" and c_J_Local_matrix is not None and c_v_circum is not None:
            shape_v = np.tile(collision.v.reshape(-1, 1), (c_v_circum.shape[0] // 6, 1))
            v_guide_current = α * c_v_circum + β * shape_v
            c_J_Local_matrix = np.c_[c_J_Local_matrix[:, :n], np.zeros((c_J_Local_matrix.shape[0], 6))]

            λ_d = 1 + (λ_adj - 1) * (r_guide - d_min) / (r_guide - d_s)
            λ_v = 1 + (λ_adj - 1) * (np.linalg.norm(collision.v[:3]) - v_min) / (v_max - v_min)
            λ_dynamic = λ_init * λ_d * λ_v

            # Apply dynamic weights
            weighted_J_Local = np.sqrt(λ_dynamic) * c_J_Local_matrix
            weighted_v_guide = λ_dynamic * v_guide_current

            # Aggregate constraints
            if J_Local is None:
                J_Local = weighted_J_Local
                v_guide = weighted_v_guide
            else:
                J_Local = np.r_[J_Local, weighted_J_Local]
                v_guide = np.r_[v_guide, weighted_v_guide]

    # Set constraints based on the current mode
    if mode == "Normal":
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v_desired.reshape((6,))
    else:
        if J_Local is not None and v_guide is not None:
            Q += (J_Local.T @ J_Local)
            c -= (v_guide.T @ J_Local).flatten()

    # Set variable bounds
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve the QP problem
    qd = qp.solve_qp(Q, c, A_in, b_in, Aeq, beq, lb=lb, ub=ub, solver="osqp")
    panda.qd[:n] = qd[:n]

    env.step(0.01)  # Update simulation
    return arrived


# Main loop to execute steps until the robot reaches the target
def run():
    arrived = False
    while not arrived:
        arrived = step_MIVG()


step_MIVG()  
run()  
