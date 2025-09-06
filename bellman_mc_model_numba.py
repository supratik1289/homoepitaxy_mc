import numpy as np
import numba as nb
from numpy import sinh, cosh, exp
import matplotlib.pyplot as plt
import time

# Simulation parameters
l_0 = 55
k_m = 5
k_p = 1
C_s = 0.01
lambda_s = 2500
lambda_p = k_p / C_s
lambda_m = k_m / C_s
Nterrace = 100
Natoms = 100000*8

# Flux function (eq. 4.8 and 4.2)
@nb.njit
def J_s(x, l_0, lambda_p, lambda_m, lambda_s):
    a = 0.5 * (l_0 - 2 * x) / lambda_s
    b = 0.5 * (l_0 + 2 * x) / lambda_s
    c = lambda_s * lambda_p
    d = lambda_s * lambda_m
    e = lambda_p * lambda_m
    f = l_0 / lambda_s
    g = lambda_s ** 2
    return -2 * (
        c * lambda_m * cosh(a) - c * lambda_m * cosh(b)
        + d * lambda_s * sinh(a) - c * lambda_s * sinh(b)
    ) / (exp(f) * (d + e + g + c) - exp(-f) * (d - e + g - c))

# Calculate probabilities (eq. 4.10)
@nb.njit
def calculate_prob(probs, flux_vals1, flux_vals2, Nterrace):
    tot_sum = 0.0
    for i in range(Nterrace):
        tot_sum += flux_vals1[i] + flux_vals2[(i + 1) % Nterrace]
    for i in range(Nterrace):
        probs[i] = (flux_vals1[i] + flux_vals2[(i + 1) % Nterrace]) / tot_sum
    # Normalize to avoid floating point drift
    s = probs.sum()
    for i in range(Nterrace):
        probs[i] /= s
    return probs, tot_sum

# Custom random choice using CDF
@nb.njit
def random_choice(probs):
    r = np.random.random()
    cum_sum = 0.0
    for i in range(probs.shape[0]):
        cum_sum += probs[i]
        if r < cum_sum:
            return i
    return probs.shape[0] - 1  # fallback

# Update step
@nb.njit
def update(terrace_widths, flux_vals1, flux_vals2, tot_sum, probs,
           l_0, lambda_p, lambda_m, lambda_s, Nterrace):
    s = random_choice(probs)

    # Update terrace widths
    terrace_widths[(s + 1) % Nterrace] += l_0 * Nterrace / 100000.0
    terrace_widths[s] -= l_0 * Nterrace / 100000.0

    # Update flux values for affected terraces
    flux_vals1[s] = J_s(terrace_widths[s], l_0, lambda_p, lambda_m, lambda_s)
    flux_vals2[s] = -J_s(-terrace_widths[s], l_0, lambda_p, lambda_m, lambda_s)
    idx = (s + 1) % Nterrace
    flux_vals1[idx] = J_s(terrace_widths[idx], l_0, lambda_p, lambda_m, lambda_s)
    flux_vals2[idx] = -J_s(-terrace_widths[idx], l_0, lambda_p, lambda_m, lambda_s)

    # Update probabilities
    probs, tot_sum = calculate_prob(probs, flux_vals1, flux_vals2, Nterrace)
    return terrace_widths, flux_vals1, flux_vals2, probs, tot_sum

# Run the simulation
@nb.njit
def simulate(terrace_widths, probs, flux_vals1, flux_vals2, tot_sum,
             Natoms, l_0, lambda_p, lambda_m, lambda_s, Nterrace):
    for _ in range(Natoms):
        terrace_widths, flux_vals1, flux_vals2, probs, tot_sum = update(
            terrace_widths, flux_vals1, flux_vals2, tot_sum, probs,
            l_0, lambda_p, lambda_m, lambda_s, Nterrace
        )
    return terrace_widths

# ---------------------------
# Initialization
# ---------------------------
terrace_widths = np.array([l_0] * Nterrace, dtype=np.float64)
flux_vals1 = np.zeros(Nterrace, dtype=np.float64)
flux_vals2 = np.zeros(Nterrace, dtype=np.float64)
for i in range(Nterrace):
    flux_vals1[i] = J_s(terrace_widths[i], l_0, lambda_p, lambda_m, lambda_s)
    flux_vals2[i] = -J_s(-terrace_widths[i], l_0, lambda_p, lambda_m, lambda_s)
probs = np.zeros(Nterrace, dtype=np.float64)
probs, tot_sum = calculate_prob(probs, flux_vals1, flux_vals2, Nterrace)

# ---------------------------
# Run simulation
# ---------------------------
start = time.time()
terrace_widths = simulate(terrace_widths, probs, flux_vals1, flux_vals2, tot_sum,
                          Natoms, l_0, lambda_p, lambda_m, lambda_s, Nterrace)
end = time.time()
print("Time taken to simulate:", end - start, "seconds")

# ---------------------------
# Plot
# ---------------------------
lengths = terrace_widths
x = np.concatenate(([0], np.cumsum(lengths)))
y = np.arange(len(lengths) + 1)

plt.figure(figsize=(10, 6))
plt.step(x, y, where='post', linewidth=2)
plt.ylim([0, 25])
plt.xlim([0, 1375])
plt.xlabel("Height (ML)")
plt.ylabel("Lateral Position")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
