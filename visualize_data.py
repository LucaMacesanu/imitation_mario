import numpy as np
import matplotlib.pyplot as plt

def generate_unique(arr):
    unique_vals, frequencies = np.unique(arr, return_counts= True)
    return unique_vals, frequencies


if __name__ == "__main__":
    #data is just a representation as of now
    action_history = np.array(['a','b','a','b','c','c','a'])
    state_history = np.array([1,2,1,2,1,1,1,2,1,1,1])
    reward_history = np.array([1,1,1,1,1,2,2,1,2,2,2])
    unique_vals, frequencies = generate_unique(action_history)
    plt.bar (unique_vals, frequencies)
    plt.show()

