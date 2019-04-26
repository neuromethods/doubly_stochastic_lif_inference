import numpy as np

def load_lif():
    """ Function that returns the used default values for the LIF model.

    :return: list
        Membrane time constant [ms], spiking threshold [mV], reset potential
        [mV], parameter for EIF (not important), membrane potential lower
        bound [mV], parameters that has to be 0 for LIF, refractory period [
        ms], membrane potential discretization [mV], index of reset
    """
    V_th = -40.
    V_lb = -150.0
    d_V = .025
    V_vec = np.arange(V_lb,V_th+d_V/2,d_V)
    V_r = -65.
    kr = np.argmin(np.abs(V_vec - V_r))  # reset index value
    VT = -50.
    DeltaT = 0.0
    Tref = 0.0
    tau_m = 10.0

    return tau_m, V_th, V_r, VT, V_lb, DeltaT, Tref, V_vec, kr