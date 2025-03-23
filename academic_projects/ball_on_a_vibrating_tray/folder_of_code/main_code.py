# -*- coding: utf-8 -*-


# ============================================================
# Importation des bibliothèques nécessaires
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.optimize import curve_fit


# ============================================================
# Paramètres globaux
# ============================================================

μ = 0.53  # Coefficient d'amortissement
g = 9.81  # Gravité (m/s^2)


# ============================================================
# Fonctions liées au mouvement du plateau
# ============================================================

def y_plate_func(w, t, A):
    """
    Fonction décrivant la position du plateau à un instant t.
    """
    return A * np.sin(w * t)


def v_plate_func(w, t, A):
    """
    Fonction décrivant la vitesse du plateau à un instant t.
    """
    return A * w * np.cos(w * t)


# ============================================================
# Fonctions liées au mouvement de la bille
# ============================================================

def y_ball_func(t, v_i, y_i):
    """
    Fonction décrivant la position de la bille à un instant t
    en chute libre, avec vitesse initiale v_i et position initiale y_i.
    """
    return -0.5 * g * t ** 2 + v_i * t + y_i


def v_ball_func(t, v_i):
    """
    Fonction décrivant la vitesse de la bille à un instant t
    en chute libre, avec vitesse initiale v_i.
    """
    return -g * t + v_i


# ============================================================
# Calculs des instants de décollage
# ============================================================

def calculate_t_decolle(w, A):
    """
    Calcule l'instant où la bille se décroche du plateau
    en fonction de la fréquence du plateau (w) et de son amplitude (A).
    """
    if w == 0:  # Cas où le plateau est immobile
        return (False,)
    else:
        K = g / (A * w ** 2)
        if K <= 1:
            return (True, np.arcsin(K) / w)
        else:
            return (False,)


def check_t_decolle_n(w, t, t_decolle):
    """
    Vérifie l'instant de décollage suivant en fonction de l'instant t actuel
    et de l'instant de décollage précédent.
    """
    n = int(t * w / (2 * np.pi))
    t_decolle_n = t_decolle[1] + n * 2 * np.pi / w
    if t_decolle_n < t:
        t_decolle_n += 2 * np.pi / w
    return t_decolle_n


# ============================================================
# Intersection
# ============================================================

def intersect(x, t, v, y, w, A):
    return y_ball_func(x - t, v, y) - y_plate_func(w, x, A)

# ============================================================
# Calcul du pas de temps
# ============================================================

def find_dt(w):
    """
    Fonction pour calculer le pas de temps optimal en fonction de la fréquence w.
    """
    if w == 0 or np.pi / (2 * w * 5) > 0.001:
        r = int(np.pi / (2 * w * 0.001))
        return np.pi / (2 * w * r)
    else:
        return np.pi / (2 * w * 5)


# ============================================================
# Fonctions pour alléger la boucle principale
# ============================================================

def update_free_motion(y_ball, v_ball, dt):
    """
    Cas où la bille est en chute libre (y_ball > y_plate).
    Met à jour la position et la vitesse de la bille.
    """
    new_y = y_ball_func(dt, v_ball[-1], y_ball[-1])
    new_v = v_ball_func(dt, v_ball[-1])
    y_ball = np.append(y_ball, new_y)
    v_ball = np.append(v_ball, new_v)
    return y_ball, v_ball


def update_sticky_regime(t, i, t_decolle, t_decolle_n, y_ball, v_ball, w, A):
    """
    Cas où la bille est en contact avec le plateau sur deux instants consécutifs.
    On vérifie si le décollage est autorisé et on met à jour les valeurs.
    """
    if t[i] > t_decolle_n:
        t[i - 1] = t_decolle_n
        y_ball[-1] = y_plate_func(w, t_decolle_n, A)
        v_ball[-1] = v_plate_func(w, t_decolle_n, A)
        y_ball_suivant = y_ball_func(0.01, v_ball[-1], y_ball[-1])
        if y_ball_suivant < y_plate_func(w, t[i - 1] + 0.01, A):
            new_y = y_plate_func(w, t[i], A)
            new_v = v_plate_func(w, t[i], A)
            t_decolle_n = check_t_decolle_n(w, t[i], t_decolle)
        else:
            new_y = y_ball_func(t[i] - t[i - 1], v_ball[-1], y_ball[-1])
            new_v = v_ball_func(t[i] - t[i - 1], v_ball[-1])
        y_ball = np.append(y_ball, new_y)
        v_ball = np.append(v_ball, new_v)
    else:
        new_y = y_plate_func(w, t[i], A)
        new_v = v_plate_func(w, t[i], A)
        y_ball = np.append(y_ball, new_y)
        v_ball = np.append(v_ball, new_v)
    return y_ball, v_ball, t_decolle_n, t


def update_collision(t, i, dt, y_ball, v_ball, t_decolle, t_decolle_n, w, μ, id_c, A):
    """
    Cas de collision entre la bille et le plateau.
    Recherche de l'instant exact de collision et mise à jour des conditions.
    """
    T_collision = bisect(lambda x: intersect(x, t[i - 2], v_ball[-2], y_ball[-2], w, A),
                         t[i - 2], t[i - 1],
                         xtol=2e-12, rtol=8.881784197001252e-16,
                         maxiter=100, disp=True)
    t[i - 1] = T_collision
    id_c.append(i - 1)
    real_v_ball = v_ball_func(T_collision - t[i - 2], v_ball[-2])
    y_ball[-1] = y_plate_func(w, T_collision, A)
    v_plate = v_plate_func(w, T_collision, A)
    v_ball[-1] = v_plate - μ * (real_v_ball - v_plate)
    y_ball_suivant = y_ball_func(0.01, v_ball[-1], y_ball[-1])

    stop = False  # Flag pour indiquer l'arrêt de la simulation
    if y_ball_suivant < y_plate_func(w, t[i - 1] + 0.01, A):
        if not t_decolle[0]:
            # La bille reste collée au plateau pour la suite : on complète et on arrête
            for k in range(i, len(t)):
                y_ball = np.append(y_ball, y_plate_func(w, t[k], A))
                v_ball = np.append(v_ball, v_plate_func(w, t[k], A))
            stop = True
        else:
            t_decolle_n = check_t_decolle_n(w, t[i - 1], t_decolle)
            if t[i] > t_decolle_n:
                t[i] = t_decolle_n
                new_y = y_plate_func(w, t_decolle_n, A)
                new_v = v_plate_func(w, t_decolle_n, A)
            else:
                new_y = y_plate_func(w, t[i], A)
                new_v = v_plate_func(w, t[i], A)
            y_ball = np.append(y_ball, new_y)
            v_ball = np.append(v_ball, new_v)
    else:
        new_y = y_ball_func(2 * dt - (T_collision - t[i - 2]), v_ball[-1], y_ball[-1])
        new_v = v_ball_func(2 * dt - (T_collision - t[i - 2]), v_ball[-1])
        y_ball = np.append(y_ball, new_y)
        v_ball = np.append(v_ball, new_v)
    return y_ball, v_ball, t, t_decolle_n, id_c, stop


# ============================================================
# Simulation principale
# ============================================================

def simulate_bouncing_ball_discret(w, t_max, A, h_i, v_i):
    """
    Simulation de la bille rebondissant sur un plateau oscillant.
    Retourne les résultats de la simulation : temps, positions et vitesses.
    """
    dt = find_dt(w)
    t_decolle = calculate_t_decolle(w, A)

    id_c = []  # Liste des indices de collision
    t_decolle_n = 0  # Instant de décollage courant
    t_absolute = np.arange(0, t_max + dt, dt)
    t = np.arange(0, t_max + dt, dt)
    y_ball = np.array([h_i], dtype=float)
    v_ball = np.array([v_i], dtype=float)

    for i in range(1, len(t)):
        # 1. Mouvement en chute libre
        if y_ball[-1] > y_plate_func(w, t[i - 1], A):
            y_ball, v_ball = update_free_motion(y_ball, v_ball, dt)
        # 2. Régime collé (la bille est en contact avec le plateau sur deux instants consécutifs)
        elif (y_ball[-1] == y_plate_func(w, t[i - 1], A)) and (y_ball[-2] == y_plate_func(w, t[i - 2], A)):
            y_ball, v_ball, t_decolle_n, t = update_sticky_regime(t, i, t_decolle, t_decolle_n, y_ball, v_ball, w, A)
        # 3. Collision
        else:
            y_ball, v_ball, t, t_decolle_n, id_c, stop = update_collision(t, i, dt, y_ball, v_ball, t_decolle,
                                                                          t_decolle_n, w, μ, id_c, A)
            if stop:
                break

    return t_absolute, y_plate_func(w, t_absolute, A), t, y_ball, v_ball, np.array(id_c)



