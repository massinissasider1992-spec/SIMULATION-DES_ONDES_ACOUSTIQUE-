# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:05:10 2026

@author: E14
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 1. PARAMÈTRES PHYSIQUES
# =========================
c0 = 340
Mach = 1.8
v = Mach * c0

theta = np.arcsin(1 / Mach)   # angle du cône de Mach
k = np.tan(theta)

# =========================
# 2. TEMPS
# =========================
dt = 0.02
steps = 120

# =========================
# 3. FIGURE 3D
# =========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_zlim(0, 250)

ax.set_title("Animation 3D du Cône de Mach (Supersonique)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z (temps / propagation)")

# =========================
# 4. CÔNE DE MACH
# =========================
def mach_cone(x0, z0, radius=80, n=40):
    """
    Génère un cône 3D autour de la trajectoire de l'avion
    """
    phi = np.linspace(0, 2*np.pi, n)
    r = np.linspace(0, radius, n)

    phi, r = np.meshgrid(phi, r)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = k * r

    # translation (position avion)
    x = x + x0
    z = z + z0

    return x, y, z

# =========================
# 5. ANIMATION
# =========================
def update(i):
    ax.clear()

    t = i * dt
    x_avion = -150 + v * t

    # avion
    ax.scatter(x_avion, 0, 0, color='red', s=60, label="Avion supersonique")

    # cône de Mach (traînée)
    for j in range(0, i, 10):
        t_j = j * dt
        x_j = -150 + v * t_j

        Xc, Yc, Zc = mach_cone(x_j, 0, radius=60)

        ax.plot_surface(
            Xc, Yc, Zc,
            color='cyan',
            alpha=0.08,
            linewidth=0,
            antialiased=True
        )

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(0, 250)

    ax.set_title(f"Cône de Mach 3D - t = {t:.2f}s")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Propagation")

    ax.legend()

# =========================
# 6. LANCEMENT
# =========================
ani = animation.FuncAnimation(fig, update, frames=steps, interval=40)

plt.show()