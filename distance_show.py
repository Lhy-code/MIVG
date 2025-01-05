import swift  # Simulation environment for robots
import spatialgeometry as sg  # For defining geometric shapes like obstacles
import spatialmath as sm  # For transformations and coordinate handling
import roboticstoolbox as rtb  # For robot modeling and control
import numpy as np  # For numerical operations
import qpsolvers as qp  # For solving QP problems
from Panda_guide import Panda_guide  # Custom Panda robot class
import matplotlib.pyplot as plt

# Initialize simulation environment
env = swift.Swift()
env.launch()

# Create Panda robot and set initial joint angles
panda = Panda_guide()
panda.q = panda.qr

# Robot configuration and parameters
n = 7
lambda_weight = 3
alpha, beta = 0.5, 0.5
r_guide = 0.08
di, ds = 0.3, 0.03

# Define obstacles
collisions = []
for i, (x, y, z, v) in enumerate([
    (0.52, 0.4, 0.3, [0.05, -0.2476, 0]),
    (0.1, 0.35, 0.65, [0, -0.2, 0]),
    (0.34, 0.3, 0.42, [0, -0.2, 0.2]),
    (0.26, 0.35, 0.62, [0, -0.283, 0]),
]):
    s = sg.Sphere(radius=0.05, pose=sm.SE3(x, y, z))
    s.v = np.array(v + [0, 0, 0])
    collisions.append(s)
    env.add(s)

# Define target
target = sg.Sphere(radius=0.02, pose=sm.SE3(0.6, -0.2, 0.0))
env.add(target)

# Add Panda robot to simulation
env.add(panda)

# Compute target pose
Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.T[:3, -1]

# Initialize distance tracking
end_effector_distances = []
obstacle_distances = {i: [] for i in range(len(collisions))}

# Prepare live plot
plt.ion()
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Predefine the x-axis (adjusted time steps)
time_steps = list(range(1201))
adjusted_time_steps = [t if t <= 600 else 600 + (t - 600) / 4 for t in time_steps]

# Set x-axis ticks and labels
xticks = []
xtick_labels = []
for t in range(0, 1201):
    if t % 50 == 0 and t <= 600:  # Before 600, every 50 steps
        xticks.append(adjusted_time_steps[t])
        xtick_labels.append(t)
    elif t > 600 and (t - 600) % 200 == 0:  # After 600, every 200 steps
        xticks.append(adjusted_time_steps[t])
        xtick_labels.append(t)

ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)

# Set axis limits
ax1.set_xlim(0, max(adjusted_time_steps))
ax1.set_ylim(0, 0.3)  # For obstacle distances
ax2.set_ylim(0, 0.5)  # For target distances

def step_lead():
    global Tep, end_effector_distances, obstacle_distances

    # Calculate robot and target positions
    Te = panda.fkine(panda.q)
    eTep = Te.inv() * Tep
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)

    # Prepare QP parameters
    Q = np.eye(n + 6)
    Q[:n, :n] *= 0.01
    Q[n:, n:] *= 1 / e
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(0.05, 0.9, n)

    Aeq, beq, Aje, b_pass = None, None, None, None
    mode = "Normal"
    for i, collision in enumerate(collisions):
        c_Ain, c_bin, d_min_c, c_Aje, c_bv, status = panda.link_collision_damper(
            collision,
            q=panda.q[:n],
            di=di,
            ds=ds,
            xi=1.0,
            r_guide=r_guide,
            start=panda.link_dict["panda_link1"],
            end=panda.link_dict["panda_hand"],
        )
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain[:, :n], np.zeros((c_Ain.shape[0], 6))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]
        obstacle_distances[i].append(d_min_c)

        if status == "Bypassing":
            mode = "Bypassing"
            if c_Aje is not None and c_bv is not None:
                shape_v = np.tile(collision.v.reshape(-1, 1), (c_bv.shape[0] // 6, 1))
                c_b_pass = alpha * c_bv + beta * shape_v
                c_Aje = np.c_[c_Aje[:, :n], np.zeros((c_Aje.shape[0], 6))]
                if Aje is None:
                    Aje, b_pass = c_Aje, c_b_pass
                else:
                    Aje = np.r_[Aje, c_Aje]
                    b_pass = np.r_[b_pass, c_b_pass]

    end_effector_distance = np.linalg.norm(Te.t - target.T[:3, -1])
    end_effector_distances.append(end_effector_distance)

    if mode == "Normal":
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v.reshape((6,))
    else:
        if Aje is not None and b_pass is not None:
            Q += lambda_weight * (Aje.T @ Aje)
            c -= lambda_weight * (b_pass.T @ Aje).flatten()

    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="osqp")
    panda.qd[:n] = qd[:n]
    env.step(0.01)

    return arrived

def update_plot():
    ax1.clear()
    ax2.clear()
    
    for i, distances in obstacle_distances.items():
        ax1.plot(adjusted_time_steps[:len(distances)], distances, label=f"Obstacle {i} Minimum Distance", linewidth=2)
    ax2.plot(adjusted_time_steps[:len(end_effector_distances)], end_effector_distances, label="End Effector Distance to Target", linestyle="--", color="black", linewidth=2)

    ax1.set_xlabel("Time Steps", fontsize=14)
    ax1.set_ylabel("Obstacle Minimum Distance (m)", fontsize=14, color="tab:blue")
    ax2.set_ylabel("End Effector Distance to Target (m)", fontsize=14, color="black")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)

    fig.canvas.draw()
    fig.canvas.flush_events()

def run():
    arrived = False
    while len(end_effector_distances) < 1201 and not arrived:
        arrived = step_lead()
        update_plot()
    print("Target reached or simulation completed.")

step_lead()
run()