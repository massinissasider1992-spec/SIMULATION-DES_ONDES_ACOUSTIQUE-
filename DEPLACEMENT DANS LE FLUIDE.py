# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:37:45 2026

@author: E14
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve

# =========================
# Paramètres physiques
# =========================
c0 = 340            # vitesse du son
Mach = 1.5
v = Mach * c0

rho0 = 1.225

# =========================
# Grille
# =========================
Nx, Ny = 200, 200
L = 100

x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]

# =========================
# Champ de pression initial
# =========================
x0 = -40

p = np.exp(-((X-x0)**2 + Y**2)/20)
p_old = p.copy()

dt = 0.0006
steps = 250

# Laplacien discret
kernel = np.array([[0,1,0],
                   [1,-4,1],
                   [0,1,0]]) / dx**2

# =========================
# Figure
# =========================
fig, ax = plt.subplots()

im = ax.imshow(p, extent=[-L/2,L/2,-L/2,L/2],
               cmap='seismic',
               vmin=-2, vmax=2,
               animated=True)

plane, = ax.plot([], [], 'ko', markersize=8)

ax.set_title("Propagation des ondes de pression dans un fluide")
ax.set_xlabel("x")
ax.set_ylabel("y")

# =========================
# Animation
# =========================
def update(i):
    global p, p_old

    # position avion
    x_av = x0 + v * i * dt
    plane.set_data([x_av], [0])

    # source perturbation (avion)
    r = np.sqrt((X-x_av)**2 + Y**2)
    source = np.exp(-r**2/8) * np.sin(i*0.3)

    # équation d'onde simplifiée
    lap = convolve(p, kernel, mode='reflect')
    p_new = 2*p - p_old + (c0*dt)**2 * lap + source*dt**2

    p_old = p
    p = p_new

    # mise à jour image
    im.set_array(p)

    return [im, plane]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=True)

plt.show()