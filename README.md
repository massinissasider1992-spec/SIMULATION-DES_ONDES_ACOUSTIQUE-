# SIMULATION-DES_ONDES_ACOUSTIQUE-
Le Code est developper au niveau de laboraoire de Mdelisation Numerique à Université de Cergy, permet de voir defferents phénomenes physique lors de deplacement  dans l'ecoulment 
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:14:39 2026

@author: E14
"""

# -*- coding: utf-8 -*-
"""
====================================================================
SIMULATION ACOUSTIQUE SUPERSONIQUE - AFFICHAGE INDIVIDUEL DE HAUTE QUALITÉ
====================================================================
"""

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# 1. PARAMÈTRES PHYSIQUES & CONFIGURATION NUMÉRIQUE
# ==========================================================
rho0 = 1.225          # Masse volumique air (kg/m³)
gamma = 1.4           # Indice adiabatique
R = 287.05            # Constante des gaz parfaits (J/kg/K)
T0 = 288.15           # Température de référence (K)
c0 = np.sqrt(gamma * R * T0)

# Configuration supersonique
Mach = 1.5
v_avion = Mach * c0
angle_mach = np.arcsin(1.0 / Mach)

# Domaine de calcul
Nx, Ny = 220, 220
Lx, Ly = 120.0, 120.0
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Source mobile
x0 = -45
sigma = 3.0
P0 = 150.0             
frequence_source = 160 

# Initialisation des matrices de pression
p_n = P0 * np.exp(-((X - x0)**2 + Y**2) / (2 * sigma**2))
p_old = p_n.copy()
p_future = np.zeros_like(p_n)

# Paramètres temporels
dt = 0.00040
Nombre_pas = 195

# Laplacien optimisé
kernel_laplacien = np.array([
    [0.25, 1.0, 0.25],
    [1.0, -5.0, 1.0],
    [0.25, 1.0, 0.25]
]) / (dx**2)

historique_pression = []
historique_energie = []
historique_resonance = []
historique_vitesse_son = []

# ==========================================================
# 2. BOUCLE DE RÉSOLUTION NUMÉRIQUE
# ==========================================================
print("Résolution de l'équation d'onde couplée non-linéaire...")

for step in range(Nombre_pas):
    temps = step * dt
    x_avion = x0 + v_avion * temps

    # Source monopolaire mobile
    R_source = np.sqrt((X - x_avion)**2 + Y**2)
    source = np.where(R_source < 2.8, P0 * np.sin(2 * np.pi * frequence_source * temps), 0)

    # Couplage non-linéaire (Pression -> Température -> Célérité)
    temperature_locale = T0 + 0.025 * np.abs(p_n)
    c_local = np.sqrt(gamma * R * temperature_locale)
    historique_vitesse_son.append(np.mean(c_local))

    # Équation d'onde discrète
    laplacien = ndimage.convolve(p_n, kernel_laplacien, mode='reflect')
    p_future = (2.0 * p_n - p_old + (c_local * dt)**2 * laplacien + source * (dt**2))

    # Amortissement des bords (Conditions absorbantes)
    mask_border = (X < -Lx/2 + 3) | (X > Lx/2 - 3) | (Y < -Ly/2 + 3) | (Y > Ly/2 - 3)
    p_future[mask_border] *= 0.90

    # Itération
    p_old = p_n.copy()
    p_n = p_future.copy()

    # Sauvegarde des données
    historique_energie.append(np.sum(p_n**2) * (dx * dy))
    historique_resonance.append(np.max(np.abs(p_n)))
    historique_pression.append(p_n.copy())

print("Simulation terminée. Génération des graphiques individuels...")

# Calcul des variables dérivées à l'état final
vitesse_particule = p_n / (rho0 * np.mean(c_local))
intensite = (p_n**2) / (rho0 * np.mean(c_local))
grad_x, grad_y = np.gradient(p_n, dx, dy)
flux_x, flux_y = -p_n * grad_x, -p_n * grad_y
flux_total = np.sqrt(flux_x**2 + flux_y**2)
acceleration = np.sqrt((grad_x / rho0)**2 + (grad_y / rho0)**2)
phase = np.arctan2(np.sin(p_n), np.cos(p_n))
t_axis = np.arange(Nombre_pas) * dt

# Configuration globale des graphiques pour une netteté maximale
plt.rcParams['figure.dpi'] = 120

# ==========================================================
# G1. NOUVELLE FIGURE : CHAMP DE PRESSION INSTANTANÉ (2D)
# ==========================================================
plt.figure(figsize=(10, 8))
im_pres = plt.imshow(p_n, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='seismic', vmin=-60, vmax=60)
plt.colorbar(im_pres, label='Pression Acoustique $p$ (Pa)')
plt.title("Champ de Pression Acoustique Instantané (Coupe 2D Z=0)", fontsize=13, fontweight='bold')
plt.xlabel("Axe X (m)")
plt.ylabel("Axe Y (m)")
plt.grid(False)
plt.show()

# ==========================================================
# G2. NOUVELLE FIGURE : PROPAGATION DE L'ONDE DANS LE FLUIDE
# ==========================================================
plt.figure(figsize=(10, 8))
# Représentation de l'onde de choc via la déformation spatiale (gradient du milieu)
gradient_fluid = np.log10(1 + np.sqrt(grad_x**2 + grad_y**2))
im_prop = plt.imshow(gradient_fluid, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='YlGnBu')
plt.colorbar(im_prop, label="Intensité de la perturbation mécanique")
plt.title("Visualisation de la Propagation de l'Onde de Choc dans le Fluide", fontsize=13, fontweight='bold')
plt.xlabel("Axe X (m)")
plt.ylabel("Axe Y (m)")
plt.grid(False)
plt.show()

# ==========================================================
# G3. INTENSITÉ ÉNERGÉTIQUE
# ==========================================================
plt.figure(figsize=(10, 8))
im1 = plt.imshow(intensite, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='inferno')
plt.colorbar(im1, label='Intensité $I$ (W/m²)')
plt.title("Carte de l'Intensité Énergétique", fontsize=13, fontweight='bold')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# ==========================================================
# G4. FLUX ÉNERGÉTIQUE AVEC VECTEURS (QUIVER)
# ==========================================================
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, flux_total, 60, cmap='plasma')
plt.colorbar(label='Module du Flux Énergétique')
sk = 7
plt.quiver(X[::sk, ::sk], Y[::sk, ::sk], flux_x[::sk, ::sk], flux_y[::sk, ::sk], color='white', scale=700)
plt.title("Flux Énergétique Acoustique et Vecteurs de Propagation", fontsize=13, fontweight='bold')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# ==========================================================
# G5. CARTE DE DÉPHASAGE
# ==========================================================
plt.figure(figsize=(10, 8))
im3 = plt.imshow(phase, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='twilight_shifted')
plt.colorbar(im3, label='Phase $\phi$ (rad)')
plt.title("Distribution Spatiale du Déphasage Acoustique", fontsize=13, fontweight='bold')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# ==========================================================
# G6. FRONTS D'ONDES (CONTOURS)
# ==========================================================
plt.figure(figsize=(9, 9))
contours = plt.contour(X, Y, p_n, levels=35, cmap='bwr', linewidths=1.2)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
plt.title("Fronts d'Ondes Sonores Géométriques", fontsize=13, fontweight='bold')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis('equal')
plt.show()

# ==========================================================
# G7. VITESSE PARTICULAIRE
# ==========================================================
plt.figure(figsize=(10, 8))
im5 = plt.imshow(vitesse_particule, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='RdBu_r')
plt.colorbar(im5, label='Vitesse particulaire $u$ (m/s)')
plt.title("Champ de Vitesse Particulaire du Fluide", fontsize=13, fontweight='bold')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# ==========================================================
# G8. ACCÉLÉRATION DES PARTICULES (MÉTHODE 3D TRIDIMENSIONNELLE)
# ==========================================================
fig_acc = plt.figure(figsize=(11, 8))
ax_acc = fig_acc.add_subplot(111, projection='3d')
surf_acc = ax_acc.plot_surface(X, Y, acceleration, cmap='viridis', edgecolor='none', antialiased=True)
ax_acc.set_title("Accélération Tridimensionnelle des Particules", fontsize=13, fontweight='bold')
ax_acc.set_xlabel("X (m)")
ax_acc.set_ylabel("Y (m)")
ax_acc.set_zlabel("Accélération ($m/s^2$)")
fig_acc.colorbar(surf_acc, ax=ax_acc, shrink=0.6)
plt.show()

# ==========================================================
# G9. HISTORIQUES TEMPORELS (Énergie, Résonance, Célérité)
# ==========================================================
plt.figure(figsize=(12, 5))
plt.plot(t_axis * 1000, historique_energie, color='#2c3e50', linewidth=2, label='Énergie')
plt.title("Évolution de l'Énergie Acoustique Totale", fontsize=12, fontweight='bold')
plt.xlabel("Temps (ms)")
plt.ylabel("Énergie")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(t_axis * 1000, historique_resonance, color='#e74c3c', linewidth=2, label='Résonance')
plt.title("Évolution de l'Amplitude de Résonance (Crête)", fontsize=12, fontweight='bold')
plt.xlabel("Temps (ms)")
plt.ylabel("Amplitude Max (Pa)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(t_axis * 1000, historique_vitesse_son, color='#16a085', linewidth=2, label='$c_{local}$')
plt.title("Évolution de la Vitesse Moyenne du Son (Effet Thermique)", fontsize=12, fontweight='bold')
plt.xlabel("Temps (ms)")
plt.ylabel("Célérité (m/s)")
plt.grid(True)
plt.show()

# ==========================================================
# G10. FENÊTRE MATPLOTLIB 3D : COUPLAGE PRESSION & CÔNE DE MACH
# ==========================================================
x_final = x0 + v_avion * Nombre_pas * dt
rayon_mach = abs(x_final - x0) * np.tan(angle_mach)
theta_c = np.linspace(0, 2*np.pi, 60)
z_c = np.linspace(0, 1, 40)
Theta_g, Z_cone_g = np.meshgrid(theta_c, z_c)

Rcone = rayon_mach * Z_cone_g
Xcone = x_final - Z_cone_g * abs(x_final - x0)
Ycone = Rcone * np.cos(Theta_g)
Zcone_plot = Rcone * np.sin(Theta_g)

fig_m3d = plt.figure(figsize=(12, 8))
ax_m3d = fig_m3d.add_subplot(111, projection='3d')
surf_p = ax_m3d.plot_surface(X, Y, p_n, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.8)
ax_m3d.plot_wireframe(Xcone, Ycone, Zcone_plot, color='black', alpha=0.2, label='Cône de Mach')
ax_m3d.set_title("Champ de Pression et Structure du Cône de Mach (Statique)", fontsize=13, fontweight='bold')
ax_m3d.set_xlabel("X (m)")
ax_m3d.set_ylabel("Y (m)")
ax_m3d.set_zlabel("Pression (Pa)")
plt.show()

# ==========================================================
# G11. INTERFACE FINALE : PLOTLY 3D INTERACTIF
# ==========================================================
fig_plotly = go.Figure()
fig_plotly.add_trace(go.Surface(x=X, y=Y, z=p_n, colorscale='Turbo', name="Pression"))
fig_plotly.add_trace(go.Surface(x=Xcone, y=Ycone, z=Zcone_plot, colorscale=[[0, 'rgba(0,0,0,0.25)'], [1, 'rgba(0,0,0,0.25)']], showscale=False, name="Cône de Mach"))
fig_plotly.update_layout(
    title="Analyse Spatiale Interactive Tridimensionnelle",
    scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Pression (Pa)'),
    width=1000, height=750
)
fig_plotly.show()
