import sys
import cma
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, isnan

sys.path.append("..")

from liion.elmodel import LiIon
from plot import Plot
from vardecoder import VarDecoder

class Context:
    def __init__(self):
        self.b = LiIon()
        self.curves = []
        self.SOC_0 = []

# Init the physical part of the decoder
the_decoder = VarDecoder()
# Individual-dependent parts
the_decoder.add_decoding_setter("C",           0.1,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("eta_c_0",    0.85,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("eta_c_i",    -0.1,  0.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("K_dif_elec",  0.0, 1e-4, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("U_0_bat",     3.0,  5.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("R_ohm_0",     0.0,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("R_ohm_SOC",  -1.0,  0.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("E_A",         0.1,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("n",           1.5,  2.5, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("A_k_00",    1e-30, 1e-7, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("K_dif_mem",   0.0,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("A_int",      -200, +200, lambda ctx: ctx.b, 8)
# Individual-independent parts
the_decoder.add_plain_setter("eta_c_T",         0.0, lambda ctx: ctx.b)
the_decoder.add_plain_setter("b_dif_elec",      0.0, lambda ctx: ctx.b)
the_decoder.add_plain_setter("T_0_dif_elec", -273.0, lambda ctx: ctx.b)
the_decoder.add_plain_setter("R_ohm_T",         0.0, lambda ctx: ctx.b)
the_decoder.add_plain_setter("b_dif_mem",       0.0, lambda ctx: ctx.b)
the_decoder.add_plain_setter("T_0_dif_mem",  -273.0, lambda ctx: ctx.b)

# Temperature is always +23C
temperature = +23.0

# Kind of infinity which is still finite
infty = 1e100

# Reading all the plots specified in the command line
plots = [Plot(filename) for filename in sys.argv[1:]]

# Patching the decoder to have SOC_0
the_decoder.add_decoding_setter("SOC_0", 0.9, 1.0, lambda ctx: ctx, len(plots))

# Perform the modelling
def compute_curves(ind):
    context = Context()
    the_decoder.decode(ind, context)
    for plot_idx in range(len(plots)):
        plot = plots[plot_idx]
        v, SOC = context.b.i_to_v(plot.time, plot.current, temperature, context.SOC_0[plot_idx])
        context.curves.append(v)
    return context

# Measure the fitness
def fitness(ind):
    context = compute_curves(ind)
    if context.b.R_ohm_0 + context.b.R_ohm_SOC < 1e-5:
        return infty
    result = 0.0
    for plot_idx in range(len(plots)):
        ref_v = plots[plot_idx].voltage
        mdl_v = context.curves[plot_idx]
        result += sqrt(sum((ref_v[i] - mdl_v[i]) ** 2 for i in range(len(ref_v))) / len(ref_v))
        if isnan(result):
            return infty
    return result

best_found = []
best_fitness = infty

# Initialize with something not too bad
while best_fitness == infty:
    best_found = the_decoder.random_individual()
    best_fitness = fitness(best_found)

# Try some more start points
for t in range(10):
    new_ind = the_decoder.random_individual()
    new_fitness = fitness(new_ind)
    if new_fitness < best_fitness:
        best_fitness = new_fitness
        best_found = new_ind

print("Starting point fitness: " + str(best_fitness))

options = {'verbose': 0, 'bounds': the_decoder.bounds()}
optimizer = cma.CMAEvolutionStrategy(list(best_found), 1.0, options)

# uses ask-tell interface to get results after each iteration
for i in range(3000):
    solutions = optimizer.ask()
    fitnesses = [fitness(x) for x in solutions]
    local_best = np.min(fitnesses)
    print("Iteration " + str(i) + ", fitness " + str(local_best))
    if local_best < best_fitness:
        best_fitness = local_best
        best_found = solutions[fitnesses.index(local_best)]
    optimizer.tell(solutions, fitnesses)

print("F = " + str(best_fitness))
the_decoder.print(best_found)
best_ctx = compute_curves(best_found)

for idx in range(len(plots)):
    plot = plots[idx]
    plt.figure(figsize=(8, 6), layout='constrained')
    plt.plot(plot.time, plot.voltage, label="data")
    plt.plot(plot.time, best_ctx.curves[idx], label="fit")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("V")
    plt.savefig("fig-" + str(idx) + ".png")
    plt.close()
