from dmp_position import PositionDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Imported from dmp_pp repo
from dmp.dmp_cartesian import DMPs_cartesian as dmp_pp

if __name__ == "__main__":
    # Load a demonstration file containing robot positions.
    demo = pd.read_csv("robot_data.csv")
    freq = 125  # Depends of the frequency of the robot
    t_step = 1 / freq
    tau = t_step * len(demo)
    print(len(demo))
    t = np.arange(0, tau, t_step)
    demo_p = demo[[
        "actual_TCP_pose_0", "actual_TCP_pose_1", "actual_TCP_pose_2", "actual_TCP_pose_3",
        "actual_TCP_pose_4", "actual_TCP_pose_5"
    ]].to_numpy()

    # Goal point
    gp_goal = np.array([-0.077089, -0.295877, 0.433379])

    N = 50  # Number of basis functions
    dmp = PositionDMP(n_bfs=N, alpha=48.0)
    dmp.train(demo_p[:, 0:3], t, tau)

    MP = dmp_pp(n_dmps=3,
                n_bfs=50,
                dt=t_step,
                x_0=None,
                x_goal=None,
                T=tau,
                K=1050,
                D=None,
                w=None,
                tol=0.0000001,
                alpha_s=1.0,
                rescale="rotodilatation",
                basis="gaussian")
    MP.imitate_path(x_des=demo_p[:, 0:3])
    MP.x_goal = gp_goal
    x_track, _, _, t_track = MP.rollout()

    # TODO: Try setting a different starting point for the dmp:
    # dmp.p0 = [x, y, z]

    # dmp.p0 = [-0.0771, -0.32, 0.45447]

    # TODO: ...or a different goal point:
    # dmp.g0 = [x, y, z]
    # dmp.g0 = [-0.077093, -0.30, 0.433387]

    dmp.gp = gp_goal

    # TODO: ...or a different time constant:
    # tau = T
    # tau = 10.0

    # Generate an output trajectory from the trained DMP
    dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)

    # Re-scale
    print(x_track.shape)
    print(dmp_p.shape)

    # 2D plot the DMP against the original demonstration
    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, demo_p[:, 0], label="Demonstration")
    axs[0].plot(t, dmp_p[:, 0], label="DMP")
    axs[0].plot(t_track, x_track[:, 0], label="DMP_revised")

    axs[0].set_xlabel("t (s)")
    axs[0].set_ylabel("X (m)")

    axs[1].plot(t, demo_p[:, 1], label="Demonstration")
    axs[1].plot(t, dmp_p[:, 1], label="DMP")
    axs[1].plot(t_track, x_track[:, 1], label="DMP_revised")
    axs[1].set_xlabel("t (s)")
    axs[1].set_ylabel("Y (m)")

    axs[2].plot(t, demo_p[:, 2], label="Demonstration")
    axs[2].plot(t, dmp_p[:, 2], label="DMP")
    axs[2].plot(t_track, x_track[:, 2], label="DMP_revised")
    axs[2].set_xlabel("t (s)")
    axs[2].set_ylabel("Z (m)")
    axs[2].legend()
    axs[2].set_xlim([0, 8])

    # 3D plot the DMP against the original demonstration
    fig2 = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label="Demonstration")
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label="DMP")
    ax.plot3D(x_track[:, 0], x_track[:, 1], x_track[:, 2], label="DMP_revised")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
