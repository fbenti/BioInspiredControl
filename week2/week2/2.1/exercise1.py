import numpy as np
import matplotlib.pyplot as plt

''' 2.1.1 '''

# ## Initialization
# # Length of simulation (time steps)
# simlen = 30
# # Target
# target = 0.0
# # Controller gain
# K = 1
# # Set delay
# delay = 0
# ## Simulation
# def simulate(target, K, delay):
#     # Output
#     y = np.zeros((simlen))
#     # Set first output
#     y[0] = 1
#     for t in range(simlen-1):
#         # Compute output
#         u = K * (target - y[t-delay])
#         y[t+1]=0.5*y[t] + 0.4*u # 1st order dynamics
#     return y

# y0 = simulate(target,0)
# y1 = simulate(target,1)
# y2 = simulate(target,2)
# y3 = simulate(target,3)

# ## Plot
# time = range(simlen)
# plt.plot(time,y0)
# plt.plot(time,y1)
# plt.plot(time,y2)
# plt.plot(time,y3)

# plt.legend(['delay = 0','delay = 1','delay = 2','delay = 3'])
# plt.xlabel('time step')
# plt.ylabel('y')
# plt.show()


''' 2.1.2 '''

## Initialization
# Length of simulation (time steps)
simlen = 30
# Target
target = 0.0
# Controller gain
K = 1
# Set delay
delay = 3
## Simulation
def simulate(target, K, delay):
    # Output
    y = np.zeros((simlen))
    # Set first output
    y[0] = 1
    for t in range(simlen-1):
        # Compute output
        u = K * (target - y[t-delay])
        y[t+1]=0.5*y[t] + 0.4*u # 1st order dynamics
    return y

y0 = simulate(target,1,delay)
y1 = simulate(target,0.5,delay)
y2 = simulate(target,0.25,delay)
y3 = simulate(target,0.1,delay)

## Plot
time = range(simlen)
plt.plot(time,y0)
plt.plot(time,y1)
plt.plot(time,y2)
plt.plot(time,y3)
plt.title("Delay = {}".format(delay))
plt.legend(['K = 1','K = 0.5','K = 0.25','K = 0.1'])
plt.xlabel('time step')
plt.ylabel('y')
plt.show()