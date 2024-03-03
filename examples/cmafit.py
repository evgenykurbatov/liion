import numpy as np

try:
    import liion.elmodel
except ImportError:
    import sys
    sys.path.append('..')
    import liion.elmodel
import importlib
importlib.reload(liion.elmodel)
from liion.elmodel import *

import csv
import cma
import sys

import matplotlib.pyplot as plt

from math import sqrt, isnan

class ConstantTemperatureBatteryDecoder:
    def __init__(self):
        # The individual stores all the battery values in a list, here are the indices
        self.IdxC = 0
        self.IdxEta0 = 1
        self.IdxEtaI = 2
        self.IdxKDifE = 3
        self.IdxU0 = 4
        self.IdxROhm0 = 5
        self.IdxROhmC = 6
        self.IdxEA = 7
        self.IdxN = 8
        self.IdxAK00 = 9
        self.IdxKDifM = 10
        self.IdxABegin = 11
        self.IdxAEnd = self.IdxABegin + 8
        self.IdxC0 = self.IdxAEnd
        self.IdxSize = self.IdxC0 + 1
        #self.IdxBDifE = ??? -- unused as T is fixed
        #self.IdxT0DifE = ??? -- unused as T is fixed
        #self.IdxBDifM = ??? -- unused as T is fixed
        #self.IdxT0DifM = ??? -- unused as T is fixed
        #self.IdxEtaT = ??? -- unused as T is fixed
        #self.IdxROhmT = ??? -- unused as T is fixed

        lower = [+10.0] * self.IdxSize
        upper = [-10.0] * self.IdxSize
        lower[self.IdxC] = 0.1
        upper[self.IdxC] = 1.0
        lower[self.IdxEta0] = 0.85
        upper[self.IdxEta0] = 1.01 # LOL
        lower[self.IdxEtaI] = -0.1
        upper[self.IdxEtaI] = +0.1
        lower[self.IdxKDifE] = 0.0
        upper[self.IdxKDifE] = 1e-4
        lower[self.IdxU0] = 3.0
        upper[self.IdxU0] = 5.0
        lower[self.IdxROhm0] = 0.0
        upper[self.IdxROhm0] = 1.0
        lower[self.IdxROhmC] = -0.1
        upper[self.IdxROhmC] = +0.1
        lower[self.IdxEA] = 0.1 # ???
        upper[self.IdxEA] = 1.0 # ???
        lower[self.IdxN] = 1.5 # ???
        upper[self.IdxN] = 2.5 # ???
        lower[self.IdxAK00] = 1e-20
        upper[self.IdxAK00] = 1e-7
        lower[self.IdxKDifM] = 0.0
        upper[self.IdxKDifM] = 0.1
        lower[self.IdxC0] = 0.8
        upper[self.IdxC0] = 1.01 # LOL
        for i in range(self.IdxABegin, self.IdxAEnd):
            lower[i] = -200.0
            upper[i] = +200.0
        self.lower = lower
        self.upper = upper

    def print(self, ind):
        print("C0 = " + str(ind[self.IdxC0]))
        print("C = " + str(ind[self.IdxC]))
        print("eta_c_0 = " + str(ind[self.IdxEta0]))
        print("eta_c_T = 0 (synthesized)")
        print("eta_c_i = " + str(ind[self.IdxEtaI]))
        print("K_dif_elec = " + str(ind[self.IdxKDifE]))
        print("b_dif_elec = 0 (synthesized)")
        print("T_0_dif_elec = -273 (synthesized)")
        print("U_0_bat = " + str(ind[self.IdxU0]))
        print("R_ohm_0 = " + str(ind[self.IdxROhm0]))
        print("R_ohm_T = 0 (synthesized)")
        print("R_ohm_SOC = " + str(ind[self.IdxROhmC]))
        print("E_A = " + str(ind[self.IdxEA]))
        print("n = " + str(ind[self.IdxN]))
        print("A_k_00 = " + str(ind[self.IdxAK00]))
        print("K_dif_mem = " + str(ind[self.IdxKDifM]))
        print("b_dif_mem = 0 (synthesized)")
        print("T_0_dif_mem = -273 (synthesized)")
        print("A_int = " + " ".join(str(t) for t in ind[self.IdxABegin : self.IdxAEnd]))

    def bounds(self):
        return self.lower, self.upper

    def random_individual(self):
        return np.random.uniform(low = self.lower, high = self.upper)

    def temperature(self, ind):
        return 23

    def initial_soc(self, ind):
        return ind[self.IdxC0]

    def decode(self, ind):
        batt = LiIon()

        batt.C = ind[self.IdxC]  # Ah

        batt.eta_c_0 = ind[self.IdxEta0]
        batt.eta_c_T = 0 #ind[self.IdxEtaT]  # deg(C)-1
        batt.eta_c_i = ind[self.IdxEtaI]  # A-1

        batt.K_dif_elec = ind[self.IdxKDifE]  # A-1
        batt.b_dif_elec = 0 #ind[self.IdxBDifE]  # deg(C)
        batt.T_0_dif_elec = -273 #ind[self.IdxT0DifE]  # deg(C)

        batt.U_0_bat = ind[self.IdxU0]  # V

        batt.R_ohm_0 = ind[self.IdxROhm0]     # Ohm
        batt.R_ohm_T = 0 #ind[self.IdxROhmT]     # Ohm K-1
        batt.R_ohm_SOC = ind[self.IdxROhmC]   # Ohm

        batt.E_A = ind[self.IdxEA]        # kJ mol-1
        batt.n = ind[self.IdxN]
        batt.A_k_00 = ind[self.IdxAK00]  # m2 s-1

        batt.K_dif_mem = ind[self.IdxKDifM]    # A-1
        batt.b_dif_mem = 0 #ind[self.IdxBDifM]    # deg(C)
        batt.T_0_dif_mem = -273 #ind[self.IdxT0DifM] # deg(C)

        batt.A_int = ind[self.IdxABegin : self.IdxAEnd]

        return batt

class Plot:
    def __init__(self, filename):
        with open(filename, "rt") as inf:
            reader = csv.DictReader(inf)
            self.time = []
            self.voltage = []
            self.current = []
            for row in reader:
                self.time.append(float(row["time"]))
                self.voltage.append(float(row["voltage"]))
                self.current.append(float(row["current"]))

the_decoder = ConstantTemperatureBatteryDecoder()
the_inf = 1e100

def build_curve(plot, ind):
    batt = the_decoder.decode(ind)
    initial_soc = the_decoder.initial_soc(ind)
    temperature = the_decoder.temperature(ind)
    v, SOC = batt.i_to_v(np.array(plot.time), np.array(plot.current), temperature, initial_soc)
    return v

def fitness(plot, ind):
    v = build_curve(plot, ind)
    reference_v = plot.voltage
    result = sqrt(sum((reference_v[i] - v[i]) ** 2 for i in range(len(v))))
    if isnan(result):
        return the_inf
    return result

plot = Plot(sys.argv[1])
best_found = []
best_fitness = the_inf

# Initialize with something not too bad
while best_fitness == the_inf:
    best_found = the_decoder.random_individual()
    best_fitness = fitness(plot, best_found)

# Try some more start points
for t in range(100):
    new_ind = the_decoder.random_individual()
    new_fitness = fitness(plot, new_ind)
    if new_fitness < best_fitness:
        best_fitness = new_fitness
        best_found = new_ind

print("Starting point fitness: " + str(best_fitness))

options = {'verbose': 0, 'bounds': [the_decoder.lower, the_decoder.upper]}
optimizer = cma.CMAEvolutionStrategy(list(best_found), 1, options)

# uses ask-tell interface to get results after each iteration
for i in range(2000):
    solutions = optimizer.ask()
    fitnesses = [fitness(plot, x) for x in solutions]
    local_best = np.min(fitnesses)
    print("Iteration " + str(i) + ", fitness " + str(local_best))
    if local_best < best_fitness:
        best_fitness = local_best
        best_found = solutions[fitnesses.index(local_best)]
    optimizer.tell(solutions, fitnesses)

print("F = " + str(best_fitness))
the_decoder.print(best_found)

plt.figure(figsize=(8, 6), layout='constrained')
plt.plot(plot.time, plot.voltage, label="data")
plt.plot(plot.time, build_curve(plot, best_found), label="fit")
plt.legend()
plt.xlabel("t [s]")
plt.ylabel("V")
plt.show()
plt.close()
