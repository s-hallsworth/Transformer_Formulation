import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


N = 15     # number of functions
T = 10    # number of time steps
steps = 20
time = np.linspace(0, T, num=steps)
dt = time[1] - time[0]  # time step

# Generate training data
def generate_data(num_samples=1000):
    X = []  # Input x(t)
    del_X = [] # Change in input x(t) - x(t-1)
    U = []  # Control u(t)
    dXdt = []
    for _ in range(num_samples):
        u = np.random.uniform(-1, 1, size=steps)
        x = np.zeros_like(u)
        del_x = np.zeros_like(u)
        x_prime_u = np.zeros_like(u)
        for t in range(1, steps):
            x_prime_u[t] = 2 * u[t-1] 
            del_x[t] = 2 * u[t-1] * dt
            x[t] = x[t-1] + 2 * u[t-1] * dt
        X.append(x)
        U.append(u)
        del_X.append(del_x)
        dXdt.append(x_prime_u)
    return np.array(X), np.array(U), np.array(del_X), np.array(dXdt)

x, u, delta_x,  x_prime_u= generate_data(num_samples=N)
print(x.shape, u.shape, delta_x.shape, x_prime_u.shape)
# x = np.zeros((N, T))
# u = np.zeros((N, T))
# x_prime_u = np.zeros((N, T))
# delta_x = np.zeros((N, T))
# time = np.linspace(0, T, num=T)


# # set initial condition: x at T=0 is 0
# for n in range(N):
#     for t in range(T):
#         u[n,t] = np.random.uniform(-1, 1, size=T)
#         x_prime_u[n,t] = u[n,t]
            

#         if t == 0:
#             continue
#         elif t < T - 1:
#             x[n,t+1] = x[n, t] + ((time[t + 1] - time[t])   * x_prime_u[n,t])
#             delta_x[n, t+1] = x[n,t+1] - x[n, t]


##Find derivative
x_prime_diff = np.zeros((N, steps))

# find derivative of x numerically
for n in range(N):
    for t in range(steps):
        if t > 0 :
            x_prime_diff[n, t] = (x[n, t] - x[n, t-1]) / dt 

print("sum(defined delta_x - numeric delta_x) = ", sum(x_prime_u - x_prime_diff))
# plt.figure(1, figsize=(12, 8))
# for i in range(N):  
#     #plt.figure(i, figsize=(12, 8))
#     # plt.plot(time, x[i, :], label=f"x (Position) from Func {i+1}")
#     # plt.plot( time, u[i, :],linestyle="dashed", label=f"u (Control Input) from Func {i+1}")
#     plt.plot( time[1:-1], x_prime_u[i, 1:-1] - x_prime_diff[i, 1:-1],"-o", label=f"num deriv. - given deriv. from Func {i+1}")
#     plt.xlabel("Time")
#     plt.ylabel("Magnitude")
#     plt.legend()
#     plt.grid(True)
# plt.show()


## Save data
ID = []
for i in range(N):
    array = (i + 1) * np.ones((steps))
    ID += [array]
    
data = {
    "time": np.tile(time, N),  # Repeat 'time' N times
    "u": u.flatten(),  # Flatten 'u' to match 'time'
    "x": x.flatten(),  # Flatten 'x' to match 'time'
    "delta_x": delta_x.flatten(),
    "ID": np.array(ID).flatten(),
}

df = pd.DataFrame(data)
file_path = f"..\\data\\data_T{T}_1.csv"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df.to_csv(file_path, index=False,mode='w+')
print("Generated data saved to ", file_path)
print(df.head())
print(df.shape)


