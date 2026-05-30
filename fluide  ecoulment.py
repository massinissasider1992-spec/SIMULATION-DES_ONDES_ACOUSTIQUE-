# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:15:22 2026

@author: E14
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# grille 3D
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)

r = np.sqrt(X**2 + Y**2)

# onde radiale simplifiée
Z = np.sin(3*r) / (r + 1)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0)

ax.set_title("Propagation onde de pression (approximation supersonique)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Amplitude")
plt.show()