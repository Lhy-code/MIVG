import swift  # Import Swift simulation environment for robot simulation
import spatialgeometry as sg  # Import spatial geometry library for defining geometric shapes (e.g., obstacles)
import spatialmath as sm  # Import spatial math library for coordinate transformation matrices
import roboticstoolbox as rtb  # Import robotics toolbox for robot modeling and control
import numpy as np  # Import Numpy for numerical calculations and array processing
import qpsolvers as qp  # Import QP solver for quadratic programming problems
from Panda_guide import Panda_lead_fix  # Import Panda_lead_fix class defined in Panda_guide

# Launch Swift simulation environment
env = swift.Swift()
env.launch()

# Create Panda robot object
panda = Panda_lead_fix()

# Set initial joint angles of Panda robot to ready position
panda.q = panda.qr

# Set the number of robot joints
n = 7

# Adjustable parameters
lambda_weight = 0.01 # Weight coefficient for velocity similarity guidance
alpha = 0.5  # Weight for bv
beta = 0.5   # Weight for shape.v
safe_distance = 0.08 # Safety distance
di=0.3 # Starting distance for obstacle damper
ds=0.03 # Ending distance for obstacle damper


# Create four obstacles with specified positions and velocities
s0 = sg.Sphere(radius=0.05, pose=sm.SE3(0.52, 0.4, 0.3))  # First obstacle at specified position
s0.v = [0.05, -0.2, 0, 0, 0, 0]  # Set velocity for first obstacle

s1 = sg.Sphere(radius=0.05, pose=sm.SE3(0.1, 0.35, 0.65))  # Second obstacle at specified position
s1.v = [0, -0.2, 0, 0, 0, 0]  # Set velocity for second obstacle

s2 = sg.Sphere(radius=0.05, pose=sm.SE3(0.34, 0.3, 0.42))  # Third obstacle at specified position
s2.v = [0, -0.2, 0.2, 0, 0, 0]  # Set velocity for third obstacle

s3 = sg.Sphere(radius=0.05, pose=sm.SE3(0.26, 0.35, 0.62))  # Fourth obstacle at specified position
s3.v = [0, -0.2, 0, 0, 0, 0]  # Set velocity for fourth obstacle
"""""
s4 = sg.Sphere(radius=0.05, pose=sm.SE3(0.26, 0.35, 0.32))  # Fifth obstacle at specified position
s4.v = [0, -0.15, 2, 0, 0, 0]  # Set velocity for fifth obstacle

s5 = sg.Sphere(radius=0.05, pose=sm.SE3(0.26, 0.35, 0.22))  # Sixth obstacle at specified position
s5.v = [0.2, -0.1, 0, 0, 0, 0]  # Set velocity for sixth obstacle

s6 = sg.Sphere(radius=0.05, pose=sm.SE3(0.26, 0.35, 0.42))  # Seventh obstacle at specified position
s6.v = [0, -0.2, 0, 0, 0, 0]  # Set velocity for seventh obstacle
"""""
# Add obstacles to collision list
collisions = [s0,s1,s2,s3]

# Create target point, represented as a small sphere
target = sg.Sphere(radius=0.02, pose=sm.SE3(0.6, -0.2, 0.0))

# Add Panda robot and obstacles to simulation environment
env.add(panda)
env.add(s0)
env.add(s1)
env.add(s2)
#env.add(s3)    
#env.add(s4)
#env.add(s5)
#env.add(s6) 
env.add(target)

Tep = panda.fkine(panda.q)  # Get initial pose of Panda end-effector
Tep.A[:3, 3] = target.T[:3, -1]  # Set target position's translation part

# Define one step of simulation control function
def step_lead():

    global Tep
    Te = panda.fkine(panda.q)  # Get current pose of Panda end-effector
    eTep = Te.inv() * Tep
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calculate desired velocity
    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)  # Calculate desired velocity

    # Initialize QP problem parameters
    Q = np.eye(n + 6)
    Q[:n, :n] *= 0.01  # Weight for joint velocity control
    Q[n:, n:] *= 1 / e  # Weight for slack variables
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]  # Maximize manipulability

    # Initialize constraints
    Aeq, beq = None, None
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(0.05, 0.9, n)

    # Add collision constraints
    Aje, b_pass = None, None
    mode = "Normal"
    for collision in collisions:
        c_Ain, c_bin, _, c_Aje, c_bv, status = panda.link_collision_damper(
            collision,
            q=panda.q[:n],
            di=di,
            ds=ds,
            xi=1.0,
            safe_distance=safe_distance,
            start=panda.link_dict["panda_link1"],
            end=panda.link_dict["panda_hand"],
        )
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain[:, :n], np.zeros((c_Ain.shape[0], 6))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

        # Check for bypassing mode
        if status == "Bypassing":
            mode = "Bypassing"

        # In bypassing mode, calculate b_pass and accumulate
        if status == "Bypassing" and c_Aje is not None and c_bv is not None:
            # Extend shape.v to be consistent with c_bv as a long column vector
            shape_v = np.tile(collision.v.reshape(-1, 1), (c_bv.shape[0] // 6, 1))  # (6 * m, 1)

            # Calculate b_pass = alpha * bv + beta * shape.v
            c_b_pass = alpha * c_bv + beta * shape_v

            # Adjust Aje column count, take first n columns and extend slack variable part
            c_Aje = np.c_[c_Aje[:, :n], np.zeros((c_Aje.shape[0], 6))]  # (6 * m, n+6)

            # Accumulate Aje and b_pass
            if Aje is None:
                Aje = c_Aje
                b_pass = c_b_pass  # Record b_pass
            else:
                Aje = np.r_[Aje, c_Aje]
                b_pass = np.r_[b_pass, c_b_pass]  # Accumulate b_pass

    if mode == "Normal":
        # In Normal mode, add equality constraint for target velocity
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v.reshape((6,))
        print("Currently executing normally")
    else:
        # In Bypassing mode, modify objective function
        if Aje is not None and b_pass is not None:
            Q += lambda_weight * (Aje.T @ Aje)
            c -= lambda_weight * (b_pass.T @ Aje).flatten()
        print("Currently executing bypass")

    # Set variable bounds
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve quadratic programming problem
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="osqp")  
    panda.qd[:n] = qd[:n]
    print("Current joint velocity:", panda.qd[:n])

    if qd is not None:
        guide_contribution = 0
        total_contribution = 0
        # Calculate velocity similarity guidance contribution
        if Aje is not None and b_pass is not None:

            # Numerator calculation
            guide_contribution = (
                qd.T @ (lambda_weight * (Aje.T @ Aje)) @ qd -
                qd.T @ (lambda_weight * (b_pass.T @ Aje).flatten())
            )
            # Denominator calculation (total objective value)
            total_contribution = qd.T @ Q @ qd + c.T @ qd

            # Calculate proportion
            guide_ratio = guide_contribution / total_contribution if total_contribution != 0 else 0
            print(f"Velocity similarity guidance proportion: {guide_ratio:.4f}")

    # Update simulation environment
    env.step(0.01)
    return arrived


# Main program, loop execute step() function until robot reaches target position
def run():
    arrived = False
    while not arrived:
        result = step_lead()


        # If task completed, exit main loop
        if isinstance(result, bool):
            arrived = result
            if arrived:
                print("Target reached.")


# Activate simulation environment
step_lead()  # Execute one control step
run()  # Start main loop