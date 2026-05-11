import numpy as np

def _omega_to_r(omega, delta):
    val = omega * delta
    return (val + np.sqrt(val**2 + 4.0)) / 2.0

resolution = "octave"
b     = 1
delta = 2**(1/2) - 2**(-1/2)
floor = -70.0
omegas = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
u_vals = np.array([0.3, 0.3, -16.1, -36.5, -54.5, floor])
r_pos = np.array([_omega_to_r(w, delta) for w in omegas])
r_neg = 1.0 / r_pos[::-1]
r_grid = np.concatenate([[0.001], r_neg[:-1], r_pos[1:], [1000.0]])
u_grid = np.concatenate([[floor], u_vals[::-1][:-1], u_vals[1:], [floor]])

print("r_grid:", r_grid)
print("u_grid:", u_grid)
print(f"r_pb_lo = r_neg[-2] = {r_neg[-2]:.6f}")
print(f"r_pb_hi = r_pos[1]  = {r_pos[1]:.6f}")
print(f"u_grid[5] = {u_grid[5]},  u_grid[6] = {u_grid[6]}")