import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


N = 5    # number of functions
T = 5    # number of time steps
steps = 10 * T
time = np.linspace(0, T, num=steps)
dt = time[1] - time[0]  # time step

# Generate training data
def generate_data(num_samples, time):
    x1_list = []  
    x2_list = []  

    for n in range(num_samples):
        v1 = 1*  np.random.rand(1)
        v2 = ( 200 * np.random.rand(1))
        
        x1 = np.zeros_like(time)
        x2 = np.zeros_like(time)
        print(x1.shape)
        for t in range(len(time)):
            print(t)
            x1[t] = (v1 * t)
            x2[t] = (v2 * t) - (0.5 * 9.81* t * t)
            
        x1_list.append(x1)
        x2_list.append(x2)
        
    return np.array(x1_list), np.array(x2_list)

x1, x2 = generate_data(num_samples=N, time=time)
print(x1.shape, x2.shape)


plt.figure(1, figsize=(12, 8))
for i in range(N):  
    plt.plot(x1[i,:], x2[i, :], "-x",label=f"Trajectory from Func {i+1}")
    plt.xlabel("distance")
    plt.ylabel("height")
    plt.legend()
    plt.grid(True)
plt.show()


## Save data
ID = []
for i in range(N):
    array = (i + 1) * np.ones((steps))
    ID += [array]
    
data = {
    "time": np.tile(time, N),  # Repeat 'time' N times
    "x1": x1.flatten(),  # Flatten 'x1' to match 'time'
    "x2": x2.flatten(),  
    "ID": np.array(ID).flatten(),
}

df = pd.DataFrame(data)
file_path = f"..\\data\\data_T{T}.csv"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df.to_csv(file_path, index=False,mode='w+')
print("Generated data saved to ", file_path)
print(df.head())
print(df.shape)


