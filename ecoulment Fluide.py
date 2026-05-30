# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:14:47 2026

@author: E14
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.linspace(-20, 20, 400)
y = np.linspace(-20, 20, 400)
X, Y = np.meshgrid(x, y)

def update(frame):
    ax.clear()
    
    x0 = -15 + frame * 0.5  # avion avance
    r = np.sqrt((X - x0)**2 + Y**2)
    
    wave = np.sin(2 * r - frame*0.3)
    
    ax.contourf(X, Y, wave, levels=50, cmap='coolwarm')
    ax.set_title("Propagation supersonique dynamique")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')

ani = animation.FuncAnimation(fig, update, frames=60, interval=50)
plt.show()
