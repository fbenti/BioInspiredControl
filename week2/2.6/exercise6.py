import numpy as np
import matplotlib.pyplot as plt

from robot import SingleLink
from cmac2 import CMAC

## Initialize simulation
Ts = 1e-2
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 300

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 30
Kv = 2

## TODO: Define parameters for periodic reference trajectory
A = np.pi
T = 5

## TODO: CMAC initialization
n_rfs = 11
xmin = [-np.pi,-np.pi]
xmax = [np.pi,np.pi]

cmac = CMAC(n_rfs, xmin, xmax, 1e-3)

## Simulation loop
for i in range(n_steps*n_trials):
    t = i*Ts
    ## TODO: Calculate the reference at this time step
    theta_ref = A*np.sin(2*np.pi*t/T)

    # Measure
    theta = plant.theta
    omega = plant.omega

    # Feedback controler
    error = (theta_ref - theta)
    tau_m = Kp * error + Kv* (-omega)

    ## TODO: Implement the CMAC controller into the loop
    tau_cmac = cmac.predict([theta, theta_ref])

    tau = tau_m  + tau_cmac
    cmac.learn(tau)
    
    # Iterate simulation dynamics
    plant.step(tau)

    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref

    print(plant.theta)
    print(theta_ref)
    print("\n")



# Plotting
# plt.plot(t_vec, theta_vec, label='theta')
# plt.plot(t_vec, theta_ref_vec, '--', label='reference')
# plt.legend()

# Plot trial error
error_vec = theta_ref_vec - theta_vec
l = int(T/Ts)
trial_error = np.zeros(n_trials)
for t in range(n_trials):
   trial_error[t] = np.sqrt(np.mean(error_vec[t*l:(t+1)*l]**2))
plt.figure()
plt.plot(trial_error)
plt.show()