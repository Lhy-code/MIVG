#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift  # 导入 Swift 仿真环境，用于机器人仿真
import spatialgeometry as sg  # 导入空间几何库，用于定义几何体（如障碍物）
import roboticstoolbox as rtb  # 导入机器人工具箱库，用于机器人建模与控制
import spatialmath as sm  # 导入空间数学库，用于处理坐标变换矩阵
import numpy as np  # 导入 Numpy，用于数值计算和数组处理
import qpsolvers as qp  # 导入 QP 求解器，用于解决二次规划问题
import time

# 启动 Swift 仿真环境
env = swift.Swift()
env.launch()

# 创建 Panda 机器人对象
panda = rtb.models.Panda()

# 设置 Panda 机器人的初始关节角度为准备状态
panda.q = panda.qr

# 设置 Panda 机器人的关节数
n = 7

# 创建三个障碍物，分别指定位置和速度
s0 = sg.Sphere(radius=0.05, pose=sm.SE3(0.52, 0.4, 0.3))  # 第一个障碍物，位于指定位置
s0.v = [0, -0.2, 0, 0, 0, 0]  # 第一个障碍物的速度设置为指定值

s1 = sg.Sphere(radius=0.05, pose=sm.SE3(0.1, 0.35, 0.65))  # 第二个障碍物，位于指定位置
s1.v = [0, -0.2, 0, 0, 0, 0]  # 第二个障碍物的速度设置为指定值,这里多了个x轴速度

s2 = sg.Sphere(radius=0.05, pose=sm.SE3(0.34, 0.3, 0.42))  # 第三个障碍物，位于指定位置
s2.v = [0, -0.2, 0.2, 0, 0, 0]  # 第三个障碍物的速度设置为指定值

s3 = sg.Sphere(radius=0.05, pose=sm.SE3(0.26, 0.15, 0.62))  # 第四个障碍物，位于指定位置
s3.v = [0, -0.2, 0, 0, 0, 0]  # 第四个障碍物的速度设置为指定值，Y轴速度不能到0.22


# 将障碍物添加到碰撞列表中
collisions = [s0, s1, s2, s3]

# 创建目标点，目标为一个小球体
target = sg.Sphere(radius=0.02, pose=sm.SE3(0.6, -0.2, 0.0))

# 将 Panda 机器人和障碍物添加到仿真环境中
env.add(panda)
env.add(s0)
env.add(s1)
env.add(s2)
env.add(s3)     
env.add(target)

# 设置目标的末端执行器位姿，使其与目标位置对齐
Tep = panda.fkine(panda.q)  # 获取 Panda 末端执行器的初始位姿
Tep.A[:3, 3] = target.T[:3, -1]  # 设置目标位置的平移部分，目标的末端位置

# 定义一步仿真控制步骤函数
def step():
    # 获取 Panda 末端执行器的当前位姿
    Te = panda.fkine(panda.q)

    # 计算当前末端执行器位姿与目标位姿之间的误差
    eTep = Te.inv() * Tep

    # 计算误差的空间误差标量，包含平移和旋转的误差
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # 计算为了使末端执行器接近目标所需的空间速度
    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)  # p_servo 用于计算末端执行器的速度

    # 定义控制增益
    Y = 0.01

    # 二次优化问题的目标矩阵 Q（n+6 维）
    Q = np.eye(n + 6)

    # 关节速度部分的目标矩阵 Q
    Q[:n, :n] *= Y  # 调整关节速度的权重

    # 松弛变量部分的目标矩阵 Q
    Q[n:, n:] = (1 / e) * np.eye(6)  # 用误差大小调整松弛变量的权重

    # 生成等式约束矩阵 Aeq 和向量 beq
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]  # 雅可比矩阵与单位矩阵拼接形成等式约束矩阵
    beq = v.reshape((6,))  # 期望的末端执行器速度（6 维）

    # 初始化不等式约束矩阵和向量
    Ain = np.zeros((n + 6, n + 6))  # 初始化不等式约束矩阵
    bin = np.zeros(n + 6)  # 初始化不等式约束向量

    # 关节的最小角度和影响角度（与关节限制阻尼相关）
    ps = 0.05  # 最小关节角度限制
    pi = 0.9  # 影响关节角度的阈值

    # 生成关节限制阻尼约束
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)  # 使用关节阻尼生成约束

    # 为每个障碍物生成碰撞限制约束，每一个障碍物生成的约束为m个，对应生成m*n的n维约束矩阵，然后叠加在一起就是l*n
    for collision in collisions:
        # 计算障碍物的碰撞约束
        c_Ain, c_bin = panda.link_collision_damper(  # 使用 link_collision_damper 计算碰撞约束
            collision,  # 当前碰撞体
            panda.q[:n],  # 当前关节配置
            0.3,  # 影响距离
            0.03,  # 停止距离
            1.0,  # 阻尼增益
            start=panda.link_dict["panda_link1"],  # 起始连杆，在xml文件可以看到，arm有7个关节，但添加了手臂部分两个，所以取前n个没问题
            end=panda.link_dict["panda_hand"],  # 末端连杆，就是在这里导致了关节数量不一致，变成了n+2
        )

        # 如果生成了有效的碰撞约束，则将其添加到不等式约束矩阵中
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain[:, :n], np.zeros((c_Ain.shape[0], 6))]  # 添加松弛变量到约束矩阵，为了使代码能跑，我取了前n列，要不然输出是n+2再加6，这样子理解是后面多的两个关节是来自于机械末端手的
            Ain = np.r_[Ain, c_Ain]  # 将新的碰撞约束堆叠到不等式约束矩阵中
            bin = np.r_[bin, c_bin]  # 将新的碰撞约束堆叠到不等式约束向量中

    # 计算目标函数的线性部分，最大化操作性
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]  # 使用负的可操作性雅可比矩阵，np.r_与np.c_分别代表行和列

    # 设置关节速度和松弛变量的上下限
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]  # 最小限值
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]  # 最大限值

    # 求解二次规划问题
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='osqp')  # 求解关节速度,正常失败都是在这一步无法求解

    # 将计算得到的关节速度应用到 Panda 机器人
    panda.qd[:n] = qd[:n]

    # 仿真环境更新
    env.step(0.01)  # 更新仿真环境，步长 0.01

    return arrived  # 返回是否到达目标


# 主程序，循环执行 step() 函数，直到机器人到达目标位置
def run():
    arrived = False
    while not arrived:
        arrived = step()


start_time=time.time()
step()  # 执行一次控制步骤
for i in range(100):
    step()  # 执行一次控制步骤
end_time=time.time()
print("average_Time:",(end_time-start_time)/100)
run()  # 启动主循环1
