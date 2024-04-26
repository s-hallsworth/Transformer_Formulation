import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

add_noise = False
N = 3  # number of functions
T = 700000
cols = ["time", "u", "x"]
x = np.zeros((N, T))
u = np.zeros((N, T))
time = np.linspace(0, 1, num=T)

x_prime_diff = np.zeros((N, T))
x_prime_u = np.zeros((N, T))
noise_x = np.random.rand(N, T)
noise_u = np.random.rand(N, T)


def func_1(t):
    u = 1 / (2 * (2 - t))
    x = t + (1 / (8 - (4 * t))) + 7 / 8
    return u, x


def func_2(t):
    u = t
    x = 1 + t + (t**3) / 3
    return u, x


def func_3(t):
    u = 2 * t
    x = 1 + t + (4 * (t**3)) / 3
    return u, x


# def func_4(t):
#     u = 10-t
#     x = t - ((10-t)**3)/3
#     return u,x

for t in range(T):
    u[0, t], x[0, t] = func_1(time[t])
    u[1, t], x[1, t] = func_2(time[t])
    u[2, t], x[2, t] = func_3(time[t])
    # u[3,t], x[3,t] = func_4(time[t])

x[:, 0] = 1  # set intial condition
# u[:,0] = 0

# Add noise
if add_noise:
    for i in range(100):
        x += np.random.rand(N, T) ** i
        u += np.random.rand(N, T) ** i

for t in range(T):
    x_prime_diff[:, t - 1] = (x[:, t] - x[:, t - 1]) / (time[t] - time[t - 1])  # dx/dt
x_prime_u = 1 + u**2

print(u.shape)
print(x_prime_diff.shape)
print(x_prime_u.shape)

print("time", time)
print("optimal x", x[0])
print("optimal u", u[0])

# Plotting
plt.figure(1, figsize=(12, 8))
markers = ["o-", "s-", "^-", "d-"]  # Different markers for each function
for i in range(N):  # [0,2]: #
    plt.plot(time, x[i, :], markers[i], label=f"x (Position) from Func {i+1}")
    plt.plot(
        time,
        u[i, :],
        markers[i],
        linestyle="dashed",
        label=f"u (Control Input) from Func {i+1}",
    )

plt.title("Position and Control Input vs. Time for " + str(N) + " Functions")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(2, figsize=(12, 8))
markers = ["o-", "s-", "^-", "d-"]  # Different markers for each function
for i in range(N):  # [0,2]: #
    plt.plot(
        time,
        x_prime_diff[i, :] - x_prime_u[i, :],
        markers[i],
        label=f"dx/dt - 1+u^2 from Func {i+1}",
    )
    # plt.plot(time, x_prime_u[i, :], markers[i], linestyle='dashed', label=f'1+u^2 from Func {i+1}')

plt.title("Calculating dx/dt - (1- u(t)**2) (for " + str(N) + " Functions)")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()

# Save data
ID = []
for i in range(N):
    array = (i + 1) * np.ones((T))
    ID += [array]
ID = np.array(ID).flatten()

data = {
    "time": np.tile(time, N),  # Repeat 'time' N times
    "u": u.flatten(),  # Flatten 'u' to match 'time'
    "x": x.flatten(),  # Flatten 'x' to match 'time'
    "ID": ID,
}

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
print(df.head())
