# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:16:10 2026

@author: E14
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paramètres physiques
# =========================
c = 340          # vitesse du son (m/s)
v = 600          # vitesse avion supersonique
Mach = v / c

# angle du cône de Mach
theta = np.arcsin(1 / Mach)

# grille
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)

# distance avion
x0 = 0

# onde simplifiée (fronts circulaires)
r = np.sqrt((X - x0)**2 + Y**2)
wave = np.sin(10 * (r - 0.2*v))

# masque cône de Mach
cone = np.abs(Y) <= np.tan(theta) * (x0 - X + 1e-6)

plt.figure(figsize=(8,6))
plt.contourf(X, Y, wave, levels=50, cmap='coolwarm', alpha=0.8)
plt.contour(X, Y, cone, levels=[0.5], colors='black', linewidths=2)

plt.title(f"Ondes de choc supersoniques (Mach = {Mach:.2f})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()