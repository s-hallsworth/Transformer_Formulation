import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
This script generates a dataset simulating trajectories of projectile motion, saves it as a CSV file, 
and plots a sample trajectory.

Main Features:
    - Simulates N trajectories over a given time interval with varying initial velocities.
    - Computes the horizontal (x1) and vertical (x2) positions for each trajectory.
    - Saves the generated data to a CSV file with columns: time, x1, x2, and ID.

Functions:
    - generate_data(num_samples, time):
        Generates the x1 and x2 trajectories for the given number of samples and time steps.

Workflow:
    1. Define simulation parameters (e.g., number of trajectories, time steps, and time increments).
    2. Call `generate_data()` to create motion trajectories and their corresponding velocities.
    3. Plot a sample trajectory for visualization.
    4. Save the trajectory data to a CSV file for further analysis.

Output:
    - A plot showing the optimal trajectory.
    - A CSV file containing the simulation data, saved in the `..\\data\\` directory.
"""


N = 300   # number of functions
T = 1   # number of time steps
incr = 0.5/19
time = np.arange(0, T,incr)
steps = len(time)
print(len(time), time)
dt = time[1] - time[0]  # time step
scale = 10


# Generate training data
def generate_data(num_samples, time):
    x1_list = []  
    x2_list = []  
    v1_list = []
    v2_list = []
    for n in range(num_samples):
        v1 =  10 * np.random.rand(1)[0]
        v2 = ( 10*np.random.rand(1))[0]
        
        v1_list += [v1]
        v2_list += [v2]
        
        x1 = np.zeros_like(time)
        x2 = np.zeros_like(time)

        for t in range(len(time)):
            #print(v1, v2, 0.5 * 9.81* time[t] * time[t])
            x1[t] = (v1 * time[t])
            x2[t] = (v2 * time[t]) - (0.5 * 9.81* time[t] * time[t])
            
        x1_list.append(x1)
        x2_list.append(x2)
    
    v_l1 = 0.2
    v_l2 = 1.5 
    opt_x1 =   v_l1 * time  
    opt_x2 =  (v_l2*time) - (0.5 * 9.81 * (time*time))
    
    x1_list[-1] = opt_x1
    x2_list[-1] = opt_x2
        
    return np.array(x1_list), np.array(x2_list), v1_list, v2_list

x1, x2, v1_list, v2_list = generate_data(num_samples=N, time=time)
print("v1 mean max min", np.mean(v1_list), max(v1_list), min(v1_list))
print("v2 mean max min", np.mean(v2_list), max(v2_list), min(v2_list))

plt.figure(1, figsize=(12, 8))
for i in [len(x1)-1]:  
    plt.plot(x1[i,:], x2[i, :], "-x",label=f"Trajectory from Func {i+1}")
    plt.xlabel("distance")
    plt.ylabel("height")
    plt.legend()
    plt.grid(True)
plt.show()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

## Save data
ID = []
for i in range(N):
    array = (i + 1) * np.ones(len(time))
    ID += [array]
    
data = {
    "time": np.tile(time, N),  # Repeat 'time' N times
    "x1": x1.flatten(),  # Flatten 'x1' to match 'time'
    "x2": x2.flatten(),  
    "ID": np.array(ID).flatten(),
}

df = pd.DataFrame(data)
rel_path = f"data\\data_T{T}.csv"
file_path = os.path.join(script_dir, rel_path)
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df.to_csv(file_path, index=False,mode='w+')
print("Generated data saved to ", file_path)
print(df.head())
print(df.shape)


