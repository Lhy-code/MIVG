import swift  
import time  
import spatialgeometry as sg  
import roboticstoolbox as rtb  
import spatialmath as sm 
import numpy as np 
import qpsolvers as qp  
from Panda_guide import Panda_guide  

env = None  # Simulation environment
Tep = None  # Target pose
collisions = []  # List of obstacles
n = 7  # Number of robot joints
v_max = -0.05  # Maximum velocity
v_min = -0.4  # Minimum velocity

def run(num_trials: int = 10, num_obstacles: int = 1):
    """
    Test three algorithms for each set of generated obstacles and record the success rate and total time.
    """
    global env, Tep, collisions

    # Map the number of obstacles to the corresponding generation function
    obstacle_generators = {
        1: generate_1obstacles_with_random_velocity,
        2: generate_2obstacles_with_random_velocity,
        3: generate_3obstacles_with_random_velocity,
        4: generate_4obstacles_with_random_velocity,
    }

    # Check if the specified number of obstacles is valid
    if num_obstacles not in obstacle_generators:
        raise ValueError(f"Invalid num_obstacles value: {num_obstacles}. Please use an integer between 1 and 4.")

    # Select the generation function based on num_obstacles
    obstacle_generator = obstacle_generators[num_obstacles]

    success_counts = {algo["name"]: 0 for algo in algorithms}
    total_times = {algo["name"]: 0.0 for algo in algorithms}  # Initialize total time record
    avg_step_times = {algo["name"]: [] for algo in algorithms}  # Record the average step times

    for trial in range(num_trials):
        print(f"\nStarting test {trial + 1}/{num_trials}...")

        # Generate the initial state of obstacles
        initial_states = obstacle_generator()

        for algo in algorithms:
            algo_name = algo["name"]
            panda = algo["panda"]
            step_func = algo["step"]

            print(f"Testing algorithm: {algo_name}")

            # Create the simulation environment
            env = swift.Swift()
            env.launch()

            # Initialize the Panda robot
            panda.q = panda.qr
            env.add(panda)

            # Reset obstacle states and add them to the simulation environment
            collisions = reset_obstacles(initial_states, env)

            # Add the target point
            target = sg.Sphere(radius=0.05, pose=sm.SE3(0.6, -0.2, 0))
            env.add(target)

            # Set the target pose for the robot's end-effector
            Tep = panda.fkine(panda.q)
            Tep.A[:3, 3] = target.T[:3, -1]

            # Record the start time for the algorithm
            algo_start_time = time.time()

            try:
                arrived = False
                step_times = []  # Record the time for each step of the algorithm
                while not arrived:
                    # Record the time for a single step
                    step_start = time.time()
                    arrived = step_func(panda)
                    step_end = time.time()
                    step_times.append(step_end - step_start)

                    if arrived == "failed":
                        break

                if arrived is True:
                    success_counts[algo_name] += 1

                # Record the average time per step for the current algorithm
                avg_step_times[algo_name].append(sum(step_times) / len(step_times))

            except Exception as e:
                print(f"Algorithm {algo_name} failed on trial {trial + 1}: {e}")
            finally:
                env.close()
                env = None

            # Record the total time
            algo_end_time = time.time()
            total_times[algo_name] += algo_end_time - algo_start_time

            # Output the current success rate and accumulated total time
            current_success_rate = success_counts[algo_name] / (trial + 1) * 100
            print(f"{algo_name} Current success rate: {current_success_rate:.2f}%, Total time: {total_times[algo_name]:.2f} seconds")

            print(f"Algorithm {algo_name} trial {trial + 1}/{num_trials} completed")

    # End of tests, output overall results
    print("\n--- Testing completed! Total trials:", num_trials, "Success rates and total times for each algorithm ---")
    for algo_name, count in success_counts.items():
        success_rate = count / num_trials * 100
        avg_time = sum(avg_step_times[algo_name]) / len(avg_step_times[algo_name]) if avg_step_times[algo_name] else 0
        print(f"{algo_name} Success rate: {success_rate:.2f}%, Average step time: {avg_time:.6f} seconds, Total time: {total_times[algo_name]:.2f} seconds")


def generate_1obstacles_with_random_velocity():  # Randomly generate one high-speed obstacle
    """
    Generate one obstacle at a fixed position with some randomness in velocity.
    """
    obstacles = []

    # Randomly generate the position
    x = np.random.uniform(0.15, 0.55)
    y = np.random.uniform(0.28, 0.4)
    z = np.random.uniform(0.3, 0.65)
    
    # First obstacle
    s0_pose = sm.SE3(x, y, z)
    s0_v = [0, np.random.uniform(-0.7, -0.2), 0, 0, 0, 0]
    obstacles.append({"pose": s0_pose, "v": s0_v})
    
    return obstacles


def generate_2obstacles_with_random_velocity():  # Generate two scattered obstacles with random velocities
    """
    Generate two obstacles at fixed positions with some randomness in velocity.
    """
    obstacles = []

    # First obstacle
    s0_pose = sm.SE3(0.52, 0.4, 0.3)
    s0_v = [0, np.random.uniform(v_min, v_max), 0, 0, 0, 0]
    obstacles.append({"pose": s0_pose, "v": s0_v})
    
    # Second obstacle
    s1_pose = sm.SE3(0.1, 0.35, 0.65)
    s1_v = [
        np.random.uniform(0.1, 0.12),  # x-axis velocity
        np.random.uniform(-0.4, -0.15),  # y-axis velocity
        0, 0, 0, 0
    ]
    obstacles.append({"pose": s1_pose, "v": s1_v})
    
    return obstacles

def generate_3obstacles_with_random_velocity():  
    """
    Generate two obstacles at fixed positions with some randomness in velocity.
    """
    obstacles = []
    
    # First obstacle
    s0_pose = sm.SE3(0.52, 0.4, 0.3)
    s0_v = [0, np.random.uniform(-0.35, -0.2), 0, 0, 0, 0]
    obstacles.append({"pose": s0_pose, "v": s0_v})
    
    # Second obstacle
    s1_pose = sm.SE3(0.1, 0.35, 0.65)
    s1_v = [
        np.random.uniform(0.1, 0.12),  
        np.random.uniform(-0.4, -0.15),  
        0, 0, 0, 0
    ]
    obstacles.append({"pose": s1_pose, "v": s1_v})
    
    # Third obstacle
    s2_pose = sm.SE3(0.34, 0.3, 0.42)
    s2_v = [
        0,
        np.random.uniform(-0.16, -0.15),  
        np.random.uniform(0.15, 0.25), 
        0, 0, 0
    ]
    obstacles.append({"pose": s2_pose, "v": s2_v})
   
    return obstacles

def generate_4obstacles_with_random_velocity():  
    """
    Generate two obstacles at fixed positions with some randomness in velocity.
    """
    obstacles = []
    
    #  First obstacle
    s0_pose = sm.SE3(0.52, 0.4, 0.3)
    s0_v = [0, np.random.uniform(-0.35, -0.2), 0, 0, 0, 0]
    obstacles.append({"pose": s0_pose, "v": s0_v})
    
    # Second obstacle
    s1_pose = sm.SE3(0.1, 0.35, 0.65)
    s1_v = [
        np.random.uniform(0.1, 0.12), 
        np.random.uniform(-0.4, -0.15),
        0, 0, 0, 0
    ]
    obstacles.append({"pose": s1_pose, "v": s1_v})
    
    # Third obstacle
    s2_pose = sm.SE3(0.34, 0.3, 0.42)
    s2_v = [
        0,
        np.random.uniform(-0.35, -0.15),  
        np.random.uniform(0.15, 0.25),  
        0, 0, 0
    ]
    obstacles.append({"pose": s2_pose, "v": s2_v})
    
    # Fourth obstacle
    s3_pose = sm.SE3(0.26, 0.35, 0.62)
    s3_v = [0, np.random.uniform(-0.25, -0.15), 0, 0, 0, 0]
    obstacles.append({"pose": s3_pose, "v": s3_v})
    
    return obstacles


def reset_obstacles(initial_states, env):
    """
    Reset the state of obstacles based on the initial states and add them to the simulation environment.
    """
    new_obstacles = []  # Store the reset obstacles

    for state in initial_states:
        # Create a new obstacle
        obstacle = sg.Sphere(radius=0.05, pose=state["pose"])
        obstacle.v = state["v"]

        # Add to the simulation environment
        env.add(obstacle)
        new_obstacles.append(obstacle)

    return new_obstacles


def step_Baseline(panda):
    # The pose of the Panda's end-effector
    Te = panda.fkine(panda.q)

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    # For each collision in the scene
    for collision in collisions:

        # Form the velocity damper inequality contraint for each collision
        # object on the robot to the collision in the scene
        c_Ain, c_bin = panda.link_collision_damper(
            collision,
            panda.q[:n],
            0.3,
            0.05,
            1.0,
            start=panda.link_dict["panda_link1"],
            end=panda.link_dict["panda_hand"],
        )

        # If there are any parts of the robot within the influence distance
        # to the collision in the scene
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain[:, :n], np.zeros((c_Ain.shape[0], 6))]

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='osqp')

    # Apply the joint velocities to the Panda
    panda.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(0.01)

    return arrived


def step_DMVG(panda):

    global env, Tep, collisions, v_max, v_min

    # Unified parameter names based on the paper
    λ_init = 3  # Initial dynamic weighting (λ_init)
    λ_adj = 2.0  # Adjustment factor for dynamic λ (λ_adj)
    α = 0.5  # Weight for guiding velocity (α)
    β = 0.5  # Weight for obstacle velocity (β)
    r_guide = 0.08  # Safe distance (effective radius of the guiding potential field)
    d_i = 0.3  # Start distance of the collision damper (d_i)
    d_s = 0.03  # End distance of the collision damper (d_s)

    Te = panda.fkine(panda.q)  # Current pose of the end-effector
    eTep = Te.inv() * Tep  # Transformation error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calculate desired velocity
    v_desired, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)

    # Initialize QP parameters
    Q = np.eye(n + 6)
    Q[:n, :n] *= 0.01  # Weight for joint velocities
    Q[n:, n:] *= 1 / e  # Slack variable weight
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]  # Maximize manipulability

    # Initialize constraints
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
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain[:, :n], np.zeros((c_Ain.shape[0], 6))]
            A_in = np.r_[A_in, c_Ain]
            b_in = np.r_[b_in, c_bin]

        # Check if in bypassing mode
        if status == "Bypassing":
            mode = "Bypassing"

        # Calculate v_guide in bypassing mode
        if status == "Bypassing" and c_J_Local_matrix is not None and c_v_circum is not None:
            shape_v = np.tile(collision.v.reshape(-1, 1), (c_v_circum.shape[0] // 6, 1))  # Expand obstacle velocity
            v_guide_current = α * c_v_circum + β * shape_v  # Compute guiding velocity

            # Adjust c_J_Local_matrix column size
            c_J_Local_matrix = np.c_[c_J_Local_matrix[:, :n], np.zeros((c_J_Local_matrix.shape[0], 6))]

            # Dynamic adjustment of λ
            λ_d = 1 + (λ_adj - 1) * (r_guide - d_min) / (r_guide - d_s)
            λ_v = 1 + (λ_adj - 1) * (np.linalg.norm(collision.v[:3]) - v_min) / (v_max - v_min)
            λ_dynamic = λ_init * λ_d * λ_v

            # Apply dynamic λ to J_Local and v_guide
            weighted_J_Local = np.sqrt(λ_dynamic) * c_J_Local_matrix
            weighted_v_guide = λ_dynamic * v_guide_current

            # Stack J_Local and v_guide
            if J_Local is None:
                J_Local = weighted_J_Local
                v_guide = weighted_v_guide
            else:
                J_Local = np.r_[J_Local, weighted_J_Local]
                v_guide = np.r_[v_guide, weighted_v_guide]

    if mode == "Normal":
        # Set equality constraints in normal mode
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v_desired.reshape((6,))
    else:
        # Modify objective function in bypassing mode
        if J_Local is not None and v_guide is not None:
            Q += (J_Local.T @ J_Local)
            c -= (v_guide.T @ J_Local).flatten()

    # Set variable bounds
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve the QP problem
    qd = qp.solve_qp(Q, c, A_in, b_in, Aeq, beq, lb=lb, ub=ub, solver="osqp")
    panda.qd[:n] = qd[:n]

    # Update simulation
    env.step(0.01)
    return arrived


panda_DMVG= Panda_guide()       # Panda Model for DMVG Algorithm
panda_default = rtb.models.Panda()  # Panda Model for Baseline Algorithm


algorithms = [
    {"name": "Baseline Algorithm", "panda": panda_default, "step": step_Baseline},
    {"name": "DMVG Algorithm", "panda": panda_DMVG, "step": step_DMVG},
]

# 示例运行
if __name__ == "__main__":
    run(num_trials=1,num_obstacles=4)