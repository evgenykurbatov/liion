import sys
import cma
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, isnan, exp

sys.path.append("..")

from liion.elmodel import LiIon
from plot import Plot
from vardecoder import VarDecoder

class Context:
    def __init__(self):
        self.b = LiIon()
        self.curves = []
        self.SOC_0 = []
        self.SOC_end = []

# Init the physical part of the decoder
the_decoder = VarDecoder()

# Individual-dependent parts
the_decoder.add_decoding_setter("C",           0.1,  3.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("eta_c_0",    0.85,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("eta_c_i",    -0.1,  0.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("K_dif_elec",  0.0, 1e-2, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("U_0_bat",     2.0, 4.35, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("R_ohm_0",     0.0,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("R_ohm_SOC",  -1.0,  0.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("E_A",         0.1,  1e4, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("A_k_00",    1e-30, 1e-2, lambda ctx: ctx.b) # encode a logarithm?
the_decoder.add_decoding_setter("K_dif_mem",   0.0,  1.0, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("A_int",      -1e1, +1e1, lambda ctx: ctx.b, 8)
the_decoder.add_decoding_setter("x_a_0",       0.0,  0.1, lambda ctx: ctx.b)
the_decoder.add_decoding_setter("x_c_1",       0.0,  0.1, lambda ctx: ctx.b)

# Individual-independent parts
the_decoder.add_plain_setter("n_cells",           1, lambda ctx: ctx.b)
the_decoder.add_plain_setter("n",               2.0, lambda ctx: ctx.b)
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
        context.SOC_end.append(SOC[-1])
    return context

# Validate the context for sanity
def constraint_violation(ctx):
    if ctx.b.R_ohm_0 + ctx.b.R_ohm_SOC < 1e-5:
        return True
    return False

cached_weights = dict()

# Generate weights, return cached values if there are some
def generate_weights(length):
    if length not in cached_weights:
        cached_weights[length] = [1 / length] * length
    return cached_weights[length]

# Measure the fitness
def fitness(ind):
    context = compute_curves(ind)
    if constraint_violation(context):
        return infty
    result = 0.0
    for plot_idx in range(len(plots)):
        ref_v = plots[plot_idx].voltage
        mdl_v = context.curves[plot_idx]
        length = len(ref_v)
        weights = generate_weights(length)
        error = sqrt(sum(weights[i] * ((ref_v[i] - mdl_v[i]) ** 2) for i in range(length)))
        first_diff = 0.2 * abs(ref_v[0] - mdl_v[0])
        last_diff = 0.2 * abs(ref_v[-1] - mdl_v[-1])
        if isnan(error) or isnan(last_diff) or context.SOC_end[plot_idx] < -0.1:
            return infty
        # we want a tighter fit at the beginning, at the end, and a good discharge
        error += first_diff + last_diff + max(0, context.SOC_end[plot_idx] - 0.5)
        result = max(result, error)
    return result

best_found = []
best_fitness = infty

# Initialize with something not too bad
while best_fitness == infty:
    best_found = the_decoder.random_individual()
    best_fitness = fitness(best_found)

# Try some more start points
for t in range(1000):
    new_ind = the_decoder.random_individual()
    new_fitness = fitness(new_ind)
    if new_fitness < best_fitness:
        best_fitness = new_fitness
        best_found = new_ind

print("Starting point fitness: " + str(best_fitness))

options = {'verbose': 0, 'bounds': the_decoder.bounds(), 'popsize': 30}
optimizer = cma.CMAEvolutionStrategy(list(best_found), 1.0, options)

# uses ask-tell interface to get results after each iteration
for i in range(1000):
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
print("C_end = " + " ".join(str(cx) for cx in best_ctx.SOC_end))

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
