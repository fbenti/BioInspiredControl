import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink

Ts = 1e-3
n_inputs = 1
n_outputs = 1
n_bases = 25
beta = 1e-5

## Initialize simulation
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 500


## TODO: Paste your experiment code from exercise 2.6
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

afc = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)

# Initialize tau feedback controller
u = 0
# Initialize error feedback controller
error_fb = 0
old_error_fb = 0

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

    # Update adaptive filter
    error_fb = error + afc.step(u,error)

    # error_fb_d = (error_fb - old_error_fb)/Ts # derivative of the error
    # old_error_fb = error_fb

    # Torque control output
    # u = Kp * error_fb + Kv * (-error_fb_d)
    u = Kp * error_fb + Kv * (-omega)    
    
    # Iterate simulation dynamics
    plant.step(u)

    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref



# Plotting
plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.legend()

# Plot trial error
error_vec = theta_ref_vec - theta_vec
l = int(T/Ts)
trial_error = np.zeros(n_trials)
for t in range(n_trials):
   trial_error[t] = np.sqrt(np.mean(error_vec[t*l:(t+1)*l]**2))
plt.figure()
plt.plot(trial_error)
plt.show()