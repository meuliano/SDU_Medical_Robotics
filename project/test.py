import pandas as pd
import numpy as np

demo = pd.read_csv("robot_data.csv")
demo2 = np.loadtxt("demo.dat", delimiter=" ", skiprows=1)

x = demo[["actual_TCP_pose_0", "actual_TCP_pose_1", "actual_TCP_pose_2"]].to_numpy()
print(x)
print(type(demo2))
