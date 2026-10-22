# -*- coding: utf-8 -*-
"""
profe/core/oraculo.py — Verdad de referencia (la raíz "correcta" de cada muestra).

El oráculo recalcula la raíz POR MUESTRA (la raíz cambia con la perturbación
de los parámetros) y sirve para comparar contra lo que devuelve cada método
del equipo.
"""
import numpy as np
from scipy.optimize import brentq


def oraculo_escalar(f, lo, hi, n_scan=4000):
    """Primera raíz (menor) con cambio de signo en [lo, hi], a 1e-13."""
    raices = oraculo_escalar_todas(f, lo, hi, n_scan=n_scan)
    return raices[0] if raices else float('nan')


def oraculo_escalar_todas(f, lo, hi, n_scan=2000):
    """TODAS las raíces con cambio de signo en [lo, hi], de menor a mayor."""
    xs = np.linspace(lo, hi, n_scan)
    try:
        ys = np.asarray(f(xs), dtype=float)
        if ys.shape != xs.shape:
            raise TypeError('f no es vectorizada')
    except (TypeError, ValueError, AttributeError):
        ys = np.array([f(v) for v in xs], dtype=float)
    raices = []
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        ya, yb = ys[i], ys[i + 1]
        if not (np.isfinite(ya) and np.isfinite(yb)):
            continue
        if ya == 0.0:
            raices.append(float(a))
            continue
        if ya * yb < 0:
            try:
                raices.append(float(brentq(f, a, b, xtol=1e-13, rtol=8.9e-16, maxiter=200)))
            except Exception:
                continue
    return raices


def oraculo_sistema(F, x0, J=None, tol=1e-13, max_iter=200):
    """Newton de alta precisión desde x0. Devuelve (vector_raiz, convergio)."""
    X = np.array(x0, dtype=float)
    n = len(X)
    h = 1e-8
    for _ in range(max_iter):
        Fx = np.array(F(X), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return X, False
        if np.max(np.abs(Fx)) < tol:
            return X, True
        Jm = np.zeros((n, n))
        for j in range(n):
            dX = np.zeros(n)
            dX[j] = h
            Jm[:, j] = (np.array(F(X + dX)) - np.array(F(X - dX))) / (2 * h)
        try:
            paso = np.linalg.solve(Jm, Fx)
        except np.linalg.LinAlgError:
            return X, False
        X = X - paso
        if not np.all(np.isfinite(X)):
            return X, False
        if np.max(np.abs(paso)) < tol:
            return X, True
    return X, False
