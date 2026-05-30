# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:49:59 2026

@author: E14
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 00:20:49 2026

@author: E14
"""

# -*- coding: utf-8 -*-
"""
====================================================================
SIMULATION 3D AVANCÉE D’ONDES ACOUSTIQUES SUPERSONIQUES
====================================================================

PHÉNOMÈNES SIMULÉS :
- Propagation d’ondes acoustiques
- Écoulement compressible
- Vitesse locale du son
- Résonance acoustique
- Déphasage
- Intensité énergétique
- Flux énergétique
- Accélération particulaire
- Fronts d’ondes
- Cône de Mach
- Animation 3D scientifique

====================================================================
"""

# ==========================================================
# IMPORTATIONS
# ==========================================================

# Importe NumPy pour la manipulation de tableaux multidimensionnels et les calculs mathématiques optimisés
import numpy as np
# Importe le module ndimage de SciPy pour effectuer des convolutions (nécessaire pour le calcul du Laplacien)
import scipy.ndimage as ndimage
# Importe l'interface pyplot de Matplotlib pour la création de graphiques et de visualisations 2D/3D
import matplotlib.pyplot as plt
# Importe le module d'animation de Matplotlib pour générer des rendus dynamiques au cours du temps
import matplotlib.animation as animation

# Importe explicitement la projection 3D pour permettre de dessiner des surfaces dans un espace à trois dimensions
from mpl_toolkits.mplot3d import Axes3D

# Importe les objets graphiques de Plotly (utilisé pour des visualisations interactives alternatives)
import plotly.graph_objects as go

# ==========================================================
# 1. PROPRIÉTÉS PHYSIQUES DE L’AIR
# ==========================================================

# Masse volumique de l'air au niveau de la mer en conditions standard (en kg/m³)
rho0 = 1.225
# Indice adiabatique de l'air (rapport des capacités thermiques Cp/Cv pour un gaz parfait diatomique)
gamma = 1.4
# Constante spécifique des gaz parfaits pour l'air sec (en J/(kg·K))
R = 287.05
# Température absolue initiale du milieu de propagation (en Kelvin, équivalent à 15°C)
T0 = 288.15

# ----------------------------------------------------------
# Vitesse initiale du son
# ----------------------------------------------------------

# Calcule la vitesse théorique du son à T0 à l'aide de la formule thermodynamique c = sqrt(gamma * R * T)
c0 = np.sqrt(gamma * R * T0)

# Affiche dans la console la valeur de la vitesse initiale calculée pour le milieu de référence
print("Vitesse initiale du son :", c0, "m/s")

# ==========================================================
# 2. PARAMÈTRES SUPERSONIQUES
# ==========================================================

# Nombre de Mach (vitesse source / vitesse son), réglé ici à 1.5 pour simuler un régime supersonique
Mach = 1.5

# Calcule la vitesse physique de l'avion en multipliant le nombre de Mach par la vitesse de base du son
v_avion = Mach * c0

# Calcule l'angle d'ouverture du cône de Mach (en radians) via la relation trigonométrique de l'onde de choc
angle_mach = np.arcsin(1 / Mach)

# Affiche la vitesse de déplacement linéaire de l'appareil (la source mobile)
print("Vitesse avion :", v_avion, "m/s")

# ==========================================================
# 3. DOMAINE NUMÉRIQUE
# ==========================================================

# Définit la résolution spatiale de la grille de calcul (180 points le long de l'axe X et 180 points sur l'axe Y)
Nx, Ny = 180, 180

# Définit les dimensions physiques totales du domaine de simulation en mètres (longueur et largeur)
Lx, Ly = 120.0, 120.0

# Génère un vecteur de Nx points uniformément répartis entre -Lx/2 et Lx/2 pour l'axe X
x = np.linspace(-Lx/2, Lx/2, Nx)
# Génère un vecteur de Ny points uniformément répartis entre -Ly/2 et Ly/2 pour l'axe Y
y = np.linspace(-Ly/2, Ly/2, Ny)

# Calcule le pas d'espace (la distance entre deux points successifs de la grille) selon l'axe X
dx = x[1] - x[0]
# Calcule le pas d'espace (la distance entre deux points successifs de la grille) selon l'axe Y
dy = y[1] - y[0]

# Crée des matrices de coordonnées 2D à partir des vecteurs 1D pour évaluer des fonctions sur toute la grille
X, Y = np.meshgrid(x, y)

# ==========================================================
# 4. SOURCE ACOUSTIQUE
# ==========================================================

# Position de départ de la source mobile sur l'axe X (décollage en périphérie gauche de la grille)
x0 = -45

# Paramètre d'étalement spatial (écart-type) de la distribution gaussienne modélisant la géométrie de la source
sigma = 3.5

# Amplitude maximale de la perturbation de pression initiale introduite par la source (en Pascals)
P0 = 120

# Fréquence temporelle d'oscillation de la source acoustique (en Hertz)
frequence_source = 150

# ==========================================================
# CHAMP INITIAL DE PRESSION
# ==========================================================

# Initialise le champ de pression à l'instant t=0 en appliquant une impulsion de forme gaussienne centrée en (x0, 0)
p_n = P0 * np.exp(
    -((X - x0)**2 + Y**2)/(2*sigma**2)
)

# Duplique le tableau p_n pour stocker l'état de la pression à l'instant précédent (t - dt), nécessaire au schéma temporel
p_old = p_n.copy()

# Alloue une matrice vide de mêmes dimensions pour accueillir les futures valeurs de pression calculées à l'instant (t + dt)
p_future = np.zeros_like(p_n)

# ==========================================================
# 5. PARAMÈTRES TEMPORELS
# ==========================================================

# Pas de temps discret de la simulation en secondes (choisi petit pour respecter la condition de stabilité CFL)
dt = 0.00045

# Nombre total de répétitions de la boucle temporelle pour faire avancer la simulation
Nombre_pas = 180

# ==========================================================
# LAPLACIEN
# ==========================================================

# Définit le noyau de convolution discret à 5 points pour estimer la dérivée seconde spatiale (Laplacien 2D)
kernel_laplacien = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
]) / dx**2

# ==========================================================
# 6. HISTORIQUES
# ==========================================================

# Initialise une liste pour mémoriser l'intégralité des matrices de pression à chaque pas de temps (pour l'animation)
historique_pression = []

# Initialise une liste pour suivre l'évolution temporelle de l'énergie acoustique totale du système
historique_energie = []

# Initialise une liste pour stocker la surpression maximale atteinte à chaque instant (analyse de résonance)
historique_resonance = []

# Initialise une liste pour enregistrer les variations de la moyenne de la vitesse du son au cours du calcul
historique_vitesse_son = []

# Affiche un message indicatif pour signaler le lancement des calculs itératifs
print("Simulation en cours...")

# ==========================================================
# 7. BOUCLE PRINCIPALE
# ==========================================================

# Démarre la boucle itérative temporelle qui va calculer les états successifs du système physique
for step in range(Nombre_pas):

    # Calcule le temps physique cumulé écoulé depuis le début de la simulation (en secondes)
    temps = step * dt

    # ------------------------------------------------------
    # Position avion
    # ------------------------------------------------------

    # Calcule la position instantanée de l'avion sur l'axe X en fonction de sa vitesse et du temps écoulé
    x_avion = x0 + v_avion * temps

    # ------------------------------------------------------
    # Source mobile
    # ------------------------------------------------------

    # Calcule la distance euclidienne entre chaque point de la grille et la position actuelle de la source mobile
    R_source = np.sqrt(
        (X - x_avion)**2 + Y**2
    )

    # Injecte une oscillation sinusoïdale de pression uniquement dans la zone restreinte entourant l'avion (rayon < 2.8)
    source = np.where(
        R_source < 2.8,
        P0 * np.sin(
            2*np.pi*frequence_source*temps
        ),
        0
    )

    # ------------------------------------------------------
    # Température locale simplifiée
    # Couplage pression-température
    # ------------------------------------------------------

    # Modélise un échauffement local du fluide induit par les fortes compressions en couplant la température à la pression absolue
    temperature_locale = (
        T0 + 0.02 * np.abs(p_n)
    )

    # ------------------------------------------------------
    # Vitesse locale du son
    # ------------------------------------------------------

    # Réévalue la vitesse locale du son en chaque point en prenant en compte la modification de la température locale
    c_local = np.sqrt(
        gamma * R * temperature_locale
    )

    # Calcule la vitesse moyenne du son sur l'ensemble de la grille de calcul à cet instant précis
    vitesse_son_moyenne = np.mean(c_local)

    # Ajoute cette valeur moyenne à la liste de suivi historique
    historique_vitesse_son.append(
        vitesse_son_moyenne
    )

    # ------------------------------------------------------
    # Laplacien
    # ------------------------------------------------------

    # Applique la convolution spatiale sur la matrice de pression courante pour obtenir sa courbure (Laplacien)
    laplacien = ndimage.convolve(
        p_n,
        kernel_laplacien,
        mode='reflect'
    )

    # ------------------------------------------------------
    # Équation d’onde compressible
    # ------------------------------------------------------

    # Résout l'équation des ondes discrétisée par un schéma aux différences finies centrées d'ordre 2 (calcul du futur état)
    p_future = (
        2*p_n
        - p_old
        + (c_local*dt)**2 * laplacien
        + source * dt**2
    )

    # ------------------------------------------------------
    # Mise à jour temporelle
    # ------------------------------------------------------

    # Écrase l'ancien état (t - dt) par l'état qui était actuel (t) pour préparer l'itération suivante
    p_old = p_n.copy()

    # Remplace l'état actuel (t) par le nouvel état calculé (t + dt)
    p_n = p_future.copy()

    # ------------------------------------------------------
    # ÉNERGIE
    # ------------------------------------------------------

    # Évalue l'énergie acoustique totale du domaine en effectuant la somme des carrés des amplitudes de pression
    energie_totale = np.sum(
        p_n**2
    )

    # Enregistre la valeur instantanée de l'énergie globale calculée dans son historique
    historique_energie.append(
        energie_totale
    )

    # ------------------------------------------------------
    # RÉSONANCE
    # ------------------------------------------------------

    # Détermine l'amplitude crête maximale absolue présente sur la grille pour suivre l'apparition de pics acoustiques
    resonance = np.max(np.abs(p_n))

    # Ajoute la valeur crête de résonance mesurée à l'historique dédié
    historique_resonance.append(
        resonance
    )

    # ------------------------------------------------------
    # HISTORIQUE PRESSION
    # ------------------------------------------------------

    # Archive une copie complète de l'état spatial de la pression pour pouvoir rejouer l'animation après la boucle
    historique_pression.append(
        p_n.copy()
    )

# Sortie de la boucle : informe l'utilisateur que l'intégration temporelle est finie
print("Simulation terminée.")

# ==========================================================
# 8. GRANDEURS PHYSIQUES
# ==========================================================

# Affiche un message de transition indiquant le début de la phase de post-traitement des données physiques
print("Calcul des grandeurs physiques...")

# ----------------------------------------------------------
# Vitesse particulaire
# ----------------------------------------------------------

# Calcule l'amplitude de la vitesse d'oscillation des particules fluides via la relation d'impédance acoustique (v = p / Z)
vitesse_particule = p_n / (rho0 * c0)

# ----------------------------------------------------------
# Intensité énergétique
# ----------------------------------------------------------

# Calcule l'intensité acoustique moyenne surfacique (puissance transportée par unité de surface en W/m²)
intensite = (
    p_n**2
) / (rho0 * c0)

# ----------------------------------------------------------
# Gradients
# ----------------------------------------------------------

# Calcule numériquement les dérivées partielles spatiales de la pression selon les directions X et Y (pentes locales)
grad_x, grad_y = np.gradient(
    p_n,
    dx,
    dy
)

# ----------------------------------------------------------
# Flux énergétique
# ----------------------------------------------------------

# Détermine la composante X du flux d'énergie (produit de la pression par l'opposé du gradient spatial)
flux_x = -p_n * grad_x

# Détermine la composante Y du flux d'énergie acoustique
flux_y = -p_n * grad_y

# Calcule la norme globale du vecteur flux énergétique (magnitude totale du transfert d'énergie en chaque point)
flux_total = np.sqrt(
    flux_x**2 + flux_y**2
)

# ----------------------------------------------------------
# Accélération
# ----------------------------------------------------------

# Calcule l'accélération particulaire locale en appliquant la loi d'Euler (le gradient de pression divisé par la masse volumique)
acceleration = np.sqrt(
    (grad_x / rho0)**2 +
    (grad_y / rho0)**2
)

# ----------------------------------------------------------
# Déphasage
# ----------------------------------------------------------

# Évalue l'argument de la fonction d'onde complexe pour cartographier la phase instantanée de l'onde entre -pi et pi
phase = np.arctan2(
    np.imag(np.exp(1j*p_n)),
    np.real(np.exp(1j*p_n))
)

# ==========================================================
# LONGUEUR D’ONDE
# ==========================================================

# Calcule la longueur d'onde spatiale nominale théorique du signal émis en utilisant la formule lambda = c / f
longueur_onde = c0 / frequence_source

# Affiche la valeur numérique de cette longueur d'onde de référence dans la console
print("Longueur d’onde :", longueur_onde)

# ==========================================================
# FRÉQUENCE DE RÉSONANCE
# ==========================================================

# Estime la fréquence fondamentale de résonance propre du domaine numérique considéré comme une cavité ouverte
frequence_resonance = c0 / (2 * Lx)

# Affiche la valeur de cette fréquence propre fondamentale exprimée en Hertz
print("Fréquence de résonance :",
      frequence_resonance, "Hz")

# ==========================================================
# 9. PRESSION 3D
# ==========================================================

# Crée une instance de figure graphique Matplotlib avec des dimensions larges adaptées à la 3D (12x8 pouces)
fig = plt.figure(figsize=(12,8))

# Ajoute un système d'axes tridimensionnels à la figure en spécifiant la projection '3d'
ax = fig.add_subplot(111, projection='3d')

# Dessine la surface 3D représentant la pression finale p_n en fonction de sa géométrie spatiale (X, Y) avec une carte de couleurs
surf = ax.plot_surface(
    X,
    Y,
    p_n,
    cmap='coolwarm',
    linewidth=0
)

# Configure le texte du titre principal du graphique de rendu de surface 3D
ax.set_title(
    "Champ de Pression Acoustique 3D"
)

# Assigne le label d'identification de l'axe géométrique horizontal X
ax.set_xlabel("X")
# Assigne le label d'identification de l'axe géométrique transversal Y
ax.set_ylabel("Y")
# Assigne le label d'identification de l'axe vertical représentant la grandeur physique mesurée (Pression)
ax.set_zlabel("Pression")

# ==========================================================
# CÔNE DE MACH
# ==========================================================

# Détermine mathématiquement l'abscisse ultime atteinte par l'avion à la fin de la durée de la simulation
x_final = x0 + v_avion * Nombre_pas * dt

# Calcule le rayon maximal théorique de la base du cône de choc à partir de l'angle de Mach et de la distance parcourue
rayon_mach = (
    abs(x_final - x0)
    * np.tan(angle_mach)
)

# Construit un échantillonnage angulaire régulier de 0 à 2*pi pour générer la section circulaire du cône
theta = np.linspace(0, 2*np.pi, 60)

# Construit un axe adimensionnel normalisé variant de 0 à 1 pour paramétrer la génératrice le long du corps du cône
z_cone = np.linspace(0, 1, 40)

# Génère le maillage bidimensionnel des coordonnées cylindriques pour dessiner la structure en entonnoir du cône
Theta, Zcone = np.meshgrid(
    theta,
    z_cone
)

# Calcule le rayon local évolutif du cône pour chaque sous-section droite le long de sa trajectoire
Rcone = rayon_mach * Zcone

# Calcule le profil des coordonnées cartésiennes X des parois géométriques du cône (pointant vers l'avion)
Xcone = (
    x_final
    - Zcone * abs(x_final - x0)
)

# Calcule les coordonnées spatiales cartésiennes Y des parois par projection trigonométrique standard
Ycone = Rcone * np.cos(Theta)

# Calcule les coordonnées de hauteur virtuelle Z pour positionner visuellement l'enveloppe du cône de Mach
Zcone_plot = Rcone * np.sin(Theta)

# Superpose le squelette filaire (wireframe) du cône théorique en noir transparent sur la nappe de pression
ax.plot_wireframe(
    Xcone,
    Ycone,
    Zcone_plot,
    color='black',
    alpha=0.25
)

# Force l'affichage immédiat à l'écran de la figure 3D construite avec sa géométrie analytique
plt.show()

# ==========================================================
# 10. VITESSE LOCALE DU SON
# ==========================================================

# Crée une nouvelle figure 2D indépendante de dimensions 10x8 pouces
plt.figure(figsize=(10,8))

# Génère une représentation sous forme d'image matricielle bidimensionnelle du champ thermique/acoustique de c_local
plt.imshow(
    c_local,
    extent=[x.min(), x.max(),
            y.min(), y.max()],
    origin='lower',
    cmap='viridis'
)

# Adjoint une barre d'échelle colorimétrique latérale indiquant précisément la correspondance des teintes avec la vitesse en m/s
plt.colorbar(
    label='Vitesse locale du son (m/s)'
)

# Définit le titre descriptif de cette cartographie de célérité locale
plt.title(
    "Propagation Locale du Son dans l’Écoulement"
)

# Ajoute l'étiquette de graduation de l'axe des abscisses (X en mètres)
plt.xlabel("X")
# Ajoute l'étiquette de graduation de l'axe des ordonnées (Y en mètres)
plt.ylabel("Y")

# Provoque l'affichage à l'écran de la carte bidimensionnelle de vitesse du son
plt.show()

# ==========================================================
# 11. INTENSITÉ ÉNERGÉTIQUE
# ==========================================================

# Initialise une nouvelle figure dédiée à l'analyse énergétique (dimensions 10x8 pouces)
plt.figure(figsize=(10,8))

# Affiche la matrice d'intensité énergétique globale en utilisant la palette de couleurs "inferno" (adaptée aux flux d'énergie)
plt.imshow(
    intensite,
    extent=[x.min(), x.max(),
            y.min(), y.max()],
    origin='lower',
    cmap='inferno'
)

# Crée la barre de légende verticale pour quantifier l'échelle d'intensité
plt.colorbar(
    label='Intensité énergétique'
)

# Assigne le titre principal de ce graphique bidimensionnel
plt.title(
    "Carte d’Intensité Énergétique"
)

# Libelle l'axe horizontal du graphique
plt.xlabel("X")
# Libelle l'axe vertical du graphique
plt.ylabel("Y")

# Déclenche l'ouverture de la fenêtre graphique d'intensité
plt.show()

# ==========================================================
# 12. FLUX ÉNERGÉTIQUE
# ==========================================================

# Ouvre une fenêtre graphique de dimensions 11x9 pouces pour tracer le comportement vectoriel du flux
plt.figure(figsize=(11,9))

# Trace des contours pleins lissés (80 niveaux) de la magnitude globale du flux à l'aide de la charte "plasma"
plt.contourf(
    X,
    Y,
    flux_total,
    80,
    cmap='plasma'
)

# Génère l'indicateur d'échelle associé au tracé de contours remplis
plt.colorbar(
    label='Flux énergétique'
)

# Variable de sous-échantillonnage pour ne pas surcharger le tracé et garder les flèches vectorielles lisibles
skip = 6

# Superpose un champ de vecteurs (quiver) représenté par des flèches blanches pointant dans le sens du déplacement de l'énergie
plt.quiver(
    X[::skip, ::skip],
    Y[::skip, ::skip],
    flux_x[::skip, ::skip],
    flux_y[::skip, ::skip],
    color='white'
)

# Spécifie l'en-tête textuel explicatif de la figure du flux acoustique
plt.title(
    "Flux Énergétique Acoustique"
)

# Identifie l'axe géométrique longitudinal
plt.xlabel("X")
# Identifie l'axe géométrique transversal
plt.ylabel("Y")

# Affiche la carte combinée contours et vecteurs fléchés
plt.show()

# ==========================================================
# 13. DÉPHASAGE
# ==========================================================

# Configure l'espace de dessin pour la cartographie des phases (10x8 pouces)
plt.figure(figsize=(10,8))

# Affiche la matrice des déphasages locaux via la table de couleurs cyclique "twilight" (idéale pour les angles périodiques)
plt.imshow(
    phase,
    extent=[x.min(), x.max(),
            y.min(), y.max()],
    origin='lower',
    cmap='twilight'
)

# Implante la barre verticale d'échelle de phase associée
plt.colorbar(
    label='Phase'
)

# Assigne le titre de la vue bidimensionnelle de phase
plt.title(
    "Carte de Déphasage"
)

# Ajoute la désignation textuelle de la dimension horizontale
plt.xlabel("X")
# Ajoute la désignation textuelle de la dimension verticale
plt.ylabel("Y")

# Rend visible le graphique de phase calculé
plt.show()

# ==========================================================
# 14. ACCÉLÉRATION
# ==========================================================

# Crée une figure dédiée pour le rendu tridimensionnel du champ d'accélération particulaire
fig_acc = plt.figure(figsize=(11,8))

# Ajoute un repère géométrique tridimensionnel (3D) à l'intérieur de cette figure
ax_acc = fig_acc.add_subplot(
    111,
    projection='3d'
)

# Génère la nappe de surface 3D représentative des variations d'accélération des particules d'air
ax_acc.plot_surface(
    X,
    Y,
    acceleration,
    cmap='viridis'
)

# Donne un titre explicite à cette représentation tridimensionnelle des forces d'accélération
ax_acc.set_title(
    "Accélération des Particules"
)

# Paramètre l'étiquette descriptive de l'axe de coordonnées X
ax_acc.set_xlabel("X")
# Paramètre l'étiquette descriptive de l'axe de coordonnées Y
ax_acc.set_ylabel("Y")
# Paramètre l'étiquette de hauteur liée à l'amplitude de l'accélération calculée
ax_acc.set_zlabel("Accélération")

# Déclenche l'affichage complet du graphe 3D de l'accélération
plt.show()

# ==========================================================
# 15. FRONTS D’ONDES
# ==========================================================

# Crée une fenêtre de tracé de dimensions confortables 12x9 pouces
plt.figure(figsize=(12,9))

# Calcule et dessine 40 lignes d'isovaleurs (lignes de niveau) pour mettre en évidence la structure géométrique des fronts d'ondes
contours = plt.contour(
    X,
    Y,
    p_n,
    levels=40,
    cmap='coolwarm'
)

# Intègre des étiquettes numériques directement incrustées sur les lignes de niveau pour indiquer les valeurs de pression locales
plt.clabel(
    contours,
    inline=True,
    fontsize=8
)

# Configure le titre explicatif supérieur de cette vue topographique des fronts d'ondes
plt.title(
    "Fronts d’Ondes Sonores"
)

# Identifie l'axe géométrique horizontal X
plt.xlabel("X")
# Identifie l'axe géométrique vertical Y
plt.ylabel("Y")

# Impose un rapport d'aspect égalitaire (1:1) pour éviter toute déformation artificielle de la forme circulaire/parabolique des ondes
plt.axis('equal')

# Affiche la carte topologique finale des fronts acoustiques
plt.show()

# ==========================================================
# 16. RÉSONANCE TEMPORELLE
# ==========================================================

# Crée une figure rectangulaire fine adaptée aux chronogrammes et aux suivis temporels unidimensionnels (10x6 pouces)
plt.figure(figsize=(10,6))

# Génère une série temporelle ordonnée (axe des abscisses) en multipliant l'indice des points par le pas élémentaire dt
temps_array = (
    np.arange(len(historique_resonance))
    * dt
)

# Trace la courbe d'évolution de la valeur maximale de résonance en fonction du temps physique écoulé avec une ligne épaisse
plt.plot(
    temps_array,
    historique_resonance,
    linewidth=2
)

# Attribue un titre à ce graphique d'analyse temporelle
plt.title(
    "Amplitude de Résonance"
)

# Libelle l'axe horizontal en spécifiant l'unité de mesure physique employée (le Temps en secondes)
plt.xlabel("Temps (s)")
# Libelle l'axe vertical pour symboliser la hauteur d'amplitude
plt.ylabel("Amplitude")

# Active le quadrillage de fond (grille) pour faciliter la lecture précise des coordonnées et des extremums
plt.grid(True)

# Affiche le chronogramme d'évolution de l'amplitude maximale de résonance
plt.show()

# ==========================================================
# 17. ÉVOLUTION VITESSE DU SON
# ==========================================================

# Ouvre une nouvelle fenêtre de tracé temporelle de dimensions 10x6 pouces
plt.figure(figsize=(10,6))

# Dessine le graphe de la vitesse du son moyenne sur le domaine en fonction de la flèche du temps
plt.plot(
    temps_array,
    historique_vitesse_son,
    linewidth=2
)

# Spécifie le texte du titre pour cette métrique thermodynamique globale
plt.title(
    "Évolution de la Vitesse du Son"
)

# Étiquette l'axe temporel horizontal
plt.xlabel("Temps (s)")
# Étiquette l'axe vertical en rappelant l'unité cinématique (m/s)
plt.ylabel("Vitesse du son (m/s)")

# Active la grille repère d'arrière-plan du graphique
plt.grid(True)

# Provoque l'affichage à l'écran de la courbe d'évolution cinématique
plt.show()

# ==========================================================
# 18. ÉNERGIE ACOUSTIQUE
# ==========================================================

# Initialise la dernière figure temporelle de dimension unitaire standard (10x6 pouces)
plt.figure(figsize=(10,6))

# Représente la courbe de décroissance/croissance de la quantité d'énergie acoustique accumulée au fil des itérations
plt.plot(
    temps_array,
    historique_energie,
    linewidth=2
)

# Fixe le titre informatif lié à l'analyse de conservation ou dissipation énergétique
plt.title(
    "Évolution de l’Énergie Acoustique"
)

# Libelle la coordonnée de temps en secondes
plt.xlabel("Temps (s)")
# Libelle la coordonnée d'ordonnée représentant la grandeur d'énergie adimensionnelle
plt.ylabel("Énergie")

# Affiche le maillage de lecture gris arrière sur le graphe d'énergie
plt.grid(True)

# Affiche la figure d'évolution de l'énergie acoustique globale
plt.show()

# ==========================================================
# 19. ANIMATION 3D
# ==========================================================

# Déclare une figure spécifique