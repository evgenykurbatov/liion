"""
Electrical model for lithium-ion barrery

Zero-frequency (i.e. quasi-stationary or slow) approximation is applied.

Reference:
[Paper] Berrueta A, Urtasun A, Ursua A, Sanchis P. Energy 144 (2018) 286-300
"""

import numpy as np
from numpy import sqrt, exp, log
import scipy as sp


class LiIon:
    """
    Attributes
    ----------
    C : float
        Battery capacity [Ah].

    A_int : array-like of float
        Interaction parameters for Redlich-Kister equation [J mol-1].
        These are 8 (eight) polynomial coefficients for `_v_int` function.

    U_0_bat : float
        Reference voltage [V].

    n : float
        Number of electrons
    A_k_00 : float
        Product of the area (of the elecrodes?) and the rate constant
        (at zero activation energy) [m2 s-1].
    """

    C = None

    n_cells = None

    eta_c_0 = None
    eta_c_T = None  # deg(C)-1
    eta_c_i = None  # A-1

    K_dif_elec   = None  # A-1
    b_dif_elec   = None  # deg(C)
    T_0_dif_elec = None  # deg(C)

    A_int   = None  # J mol-1

    U_0_bat = None  # V

    R_ohm_0   = None  # Ohm
    R_ohm_T   = None  # Ohm K-1
    R_ohm_SOC = None  # Ohm

    E_A     = None  # kJ mol-1
    n       = None
    A_k_00  = None  # m2 s-1

    K_dif_mem   = None  # A-1
    b_dif_mem   = None  # deg(C)
    T_0_dif_mem = None  # deg(C)

    x_a_0    = None
    x_c_1    = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


    def i_to_v(self, t, i, T, SOC_0):
        """
        Parameters
        ----------
        t : ndarray of floats
            Time points [s].
        i : ndarray of floats
            Current at the time points `t` [A].
        T : float or ndarray of floats
            Temperature [deg(C)]. May be constant or ndarray of the shape of `i`.
        SOC_0 : float, optional
            State of charge at the initial time. Between 0 and 1.

        Returns
        -------
        ndarray of floats
            Battery's voltage at the time points `t` [V].
        ndarray of floats
            SOC, state of charge at the time points `t`.
        """

        # Efficiency
        eta_c = _eta_c(i, T, self.eta_c_0, self.eta_c_T, self.eta_c_i)
        self.eta_c = eta_c

        # State of charge
        SOC = _SOC(t, i, eta_c, 3600*self.C, SOC_0)
        self.SOC = SOC

        # Electrode diffusion
        R_dif_elec = _R_dif_elec(T, self.K_dif_elec, self.b_dif_elec, self.T_0_dif_elec)
        self.R_dif_elec = R_dif_elec

        # State of charge corresponding to the lithium molar fraction
        # at the electrode surface
        SOC_sur = _SOC_sur(i, eta_c, R_dif_elec, SOC)
        self.SOC_sur = SOC_sur

        # Molar fractions
        x_a, x_c = _xx(SOC_sur, self.x_a_0, self.x_c_1)
        self.x_a = x_a
        self.x_c = x_c

        # Contribution to equilibrium voltage due to non-ideal interactions
        v_int = _v_int(x_a, x_c, self.A_int)
        self.v_int = v_int

        # Equilibrium voltage
        v_eq = _v_eq(T, x_a, x_c, v_int, self.U_0_bat, self.n_cells)
        self.v_eq = v_eq

        # Ohmic phenomena
        R_ohm = _R_ohm(T, SOC, self.R_ohm_0, self.R_ohm_T, self.R_ohm_SOC)
        self.R_ohm = R_ohm

        # Activation potential
        v_act = _v_act(i, T, x_a, x_c, 1e3*self.E_A, self.n, self.A_k_00)
        self.v_act = v_act

        # Faraday current
        i_F = _i_F(T, x_a, x_c, v_act, 1e3*self.E_A, self.n, self.A_k_00)
        self.i_F = i_F

        # WRONG! Don't use it!
        # Charge-transfer resistance
        R_ct = _R_ct(T, x_a, x_c, 1e3*self.E_A, self.n, self.A_k_00)
        self.R_ct = R_ct

        # Membrane diffusion
        R_dif_mem = _R_dif_mem(T, self.K_dif_mem, self.b_dif_mem, self.T_0_dif_mem)
        self.R_dif_mem = R_dif_mem

        #
        # Result

        # Voltage on the battery
        v = v_eq - v_act - i*(R_ohm + R_dif_mem)

        return v, SOC



def _eta_c(i, T, eta_c_0, eta_c_T, eta_c_i):
    return eta_c_0 + eta_c_T*T + eta_c_i*i


def _SOC(t, i, eta_c, C, SOC_0):
    return SOC_0 + sp.integrate.cumtrapz(-eta_c*i/C, t, initial=0)


def _SOC_sur(i, eta_c, R_dif_elec, SOC):
    return SOC - eta_c*i*R_dif_elec


def _xx(SOC_sur, x_a_0, x_c_1):
    x_a = x_a_0 + (1 - x_a_0) * SOC_sur
    x_c = x_c_1 + (1 - x_c_1) * (1 - SOC_sur)
    return x_a, x_c


def _v_int(x_a, x_c, A):
    # It is not known exactly how the variables `x_a` and `x_c` were combined,
    # according to the Paper. The authors concealed this.

    """
    x = sqrt(x_a*x_c)
    res = np.array([ A[k]*( (2*x-1)**(k+1) - 2*x*k*(1-x)/(2*x-1)**(1-k) )
                     for k in range(8) ]).sum(axis=0)
    return res
    """

    # Faraday constant
    F = 9.649e4    # C mol-1

    res_a = np.array([ A[k]*( (2*x_a-1)**(k+1) - 2*x_a*k*(1-x_a)/(2*x_a-1)**(1-k) )
                       for k in range(8) ]).sum(axis=0)
    res_c = np.array([ A[k]*( (2*x_c-1)**(k+1) - 2*x_c*k*(1-x_c)/(2*x_c-1)**(1-k) )
                       for k in range(8) ]).sum(axis=0)

    return (res_c - res_a) / F


def _v_eq(T, x_a, x_c, v_int, U_0_bat, n_cells):
    # Universal gas constant
    R_td = 8.314   # J mol-1 K-1
    # Fermi constant
    F = 9.649e4    # C mol-1

    return U_0_bat + n_cells * R_td*(273+T)/F * log((1-x_c)*x_a/(x_c*(1-x_a))) + v_int


def _R_ohm(T, SOC, R_ohm_0, R_ohm_T, R_ohm_SOC):
    return R_ohm_0 + R_ohm_T*(273+T) + R_ohm_SOC*SOC


def _v_act(i, T, x_a, x_c, E_A, n, A_k_00):
    # Universal gas constant
    R_td = 8.314   # J mol-1 K-1
    # Faraday constant
    F = 9.649e4    # C mol-1

    return 2.0/(n*F) * ( E_A + R_td*(273+T) * log(i/(F*A_k_00) / sqrt(x_a*x_c)) )


def _i_F(T, x_a, x_c, v_act, E_A, n, A_k_00):
    # Universal gas constant
    R_td = 8.314   # J mol-1 K-1
    # Faraday constant
    F = 9.649e4    # C mol-1

    return sqrt(x_a*x_c) * F*A_k_00 * exp( (-E_A + 0.5*n*F*v_act) / (R_td*(273+T)) )


def _R_ct(T, x_a, x_c, E_A, n, A_k_00):
    # WRONG!
    # Universal gas constant
    R_td = 8.314   # J mol-1 K-1
    # Faraday constant
    F = 9.649e4    # C mol-1

    return R_td*(273+T)/(n*F**2*A_k_00) * exp(E_A/(R_td*(273+T))) / sqrt(x_a*x_c)


def _R_dif_elec(T, K_dif_elec, b_dif_elec, T_0_dif_elec):
    return K_dif_elec * exp(b_dif_elec/(T - T_0_dif_elec))


def _R_dif_mem(T, K_dif_mem, b_dif_mem, T_0_dif_mem):
    return K_dif_mem * exp(b_dif_mem/(T - T_0_dif_mem))
