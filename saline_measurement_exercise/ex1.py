"""_summary_
This is a simple example of a python script."""
import numpy as np
import pandas as pd

PATH = "prelim_saline_data/"
data = pd.read_csv(PATH + "0.1%_1mm_depth_5_samples.txt")
x = [1e3, 2e3, 3e3, 7e3, 11e3, 17e3, 23e3, 31e3, 43e3, 61e3, 89e3, 127e3, 179e3, 251e3, 349e3]
print(x)
print(data[0])
