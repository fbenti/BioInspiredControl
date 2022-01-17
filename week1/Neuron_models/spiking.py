import numpy as np

def lif(u_m, I, Ts, deltaT):

    tau_m = Rm * Cm
    it = 0
    while it < Ts:
        # Calculate derivative
        du_dt = (u_rest - u_m + Rm*I) / tau_m
        # Update voltage
        u_m += du_dt*deltaT
        # Evaluate membrane potential
        if u_m > u_thresh:
            u_m = u_rest
        # Update simulation time
        it += deltaT
    return u_m

if __name__ == "__main__":

    # Circuit parameter
    Rm = 10e6
    Cm = 1e-9
    u_thresh = -50e-3
    u_rest = -65e-3 


    # Initialize a membrane potential variable u_m and a time variable to zero
    u_m = 0
    t = 0

    # Simulation time step
    deltaT = 1e-5

    # 1.4.2 
    I = 1e-9
    # Time simulation
    Ts = 100e-3

    print(lif(u_m,I,Ts,deltaT))

   


