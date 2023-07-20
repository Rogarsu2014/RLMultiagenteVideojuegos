import numpy as np

num_bins = [10, 10, 10, 10]
# Initialize the Q-table with zeros
q_table = np.zeros((*num_bins, 2))

print(q_table)

