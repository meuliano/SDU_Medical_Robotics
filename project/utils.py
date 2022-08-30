import numpy as np
import pandas as pd

# Read impedeance data from file, replace i to j to convert the string to a complex number later on
data = pd.read_csv("test02.csv").replace("i", "j", regex=True)
# Input frequency data in Hz
frequency = [
    1e3, 2e3, 3e3, 7e3, 11e3, 17e3, 23e3, 31e3, 43e3, 61e3, 89e3, 127e3, 179e3, 251e3, 349e3
]
# Pose of the grid point in the x, y, z direction
# We will use the averaged value of z and assume that z is constant
grid_data = pd.read_csv("grid_points.csv").to_numpy()
print(grid_data.shape)

# Take the 5 highest impedances
frequency_refined = frequency[-5:]
resistence = data.iloc[:, [-5, -4, -3, -2, -1]].to_numpy().astype(np.complex)

# print(resistence[:, 1].imag)
# print(frequency_refined)