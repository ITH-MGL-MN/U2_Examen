# -*- coding: utf-8 -*-
"""
profe/core/modelos.py — Modelo numérico de cada caso.

Construye, para un tema dado, la función `f(x, **params)` (o el sistema
`F(X, vals)`) con `sympy` + `lambdify`, más su derivada/Jacobiano y los
metadatos. Es el corazón del análisis gráfico y de la simulación de Monte
Carlo.
"""
import warnings

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from .catalogo import CATALOGO

# Nombre real de la incógnita en cada expresión escalar (evita ambigüedad de
# símbolos). T6 usa 'TA' (tensión en el cable) para no chocar con 'T'
# (temperatura) de T4/T11; T10 y T17 usan 'omega' para que el nombre ASCII no
# choque con 'w' de T6.
_VAR_REAL = {'T1': 'x', 'T2': 'd', 'T3': 'th', 'T4': 'T', 'T5': 'VD', 'T6': 'TA', 'T7': 'H',
             'T8': 'th', 'T9': 'R', 'T10': 'omega', 'T11': 'v', 'T12': 'Rr', 'T13': 'f',
             'T16': 'v', 'T17': 'omega', 'T18': 'th'}

_LATEX = {
    'x': 'x', 'd': 'd', 'th': r'\theta', 'T': 'T', 'VD': 'V_D', 'TA': 'T_A',
    'H': 'H', 'R': 'R', 'Rr': 'R', 'f': 'f', 'v': 'v', 'omega': r'\omega',
    'th1': r'\theta_1', 'th2': r'\theta_2', 'Q1': 'Q_1', 'Q2': 'Q_2',
    'k1': 'k_1', 'k2': 'k_2', 'm': 'm', 'g': 'g', 'h': 'h', 'L': 'L', 'a': 'a',
    'q': 'q', 'Q': 'Q', 'e0': r'\varepsilon_0', 'objetivo': '\\text{objetivo}',
    'IS': 'I_S', 'VT': 'V_T', 'VCC': 'V_{CC}', 'RL': 'R_L', 'n': 'n', 'w': 'w',
    'y': 'y', 'y0': 'y_0', 'S': 'S', 'B': 'B', 'nn': 'n', 'v0': 'v_0',
    'C': 'C', 't': 't', 'qq': 'q/q_0', 'Z': 'Z', 'p': 'p', 'R2': 'R',
    'Xaf': 'X_{Af}', 'eps': r'\epsilon', 'Re': 'Re', 'rho': r'\rho',
    'rhos': r'\rho_s', 'mu': r'\mu', 'wn': r'\omega_n', 'z': r'\zeta',
    'F0': 'F_0', 'A': 'A', 'Tb': 'T_b', 'To': 'T_o', 'c': 'c', 'Ws': 'W_S',
    'L1': 'L_1', 'L2': 'L_2', 'd1': 'd_1', 'd2': 'd_2', 'K1': 'K_1',
    'K2': 'K_2', 'K3': 'K_3', 'dH': r'\Delta H',
}


class _Symb(sp.Symbol):
    """Símbolo SymPy con nombre LaTeX propio (para imprimir bonito)."""

    def __new__(cls, name, latex_name=None, **assumptions):
        obj = sp.Symbol.__new__(cls, name, **assumptions)
        obj._latex_name = latex_name
        return obj

    def _latex(self, printer=None):
        return self._latex_name or super()._latex(printer)


def _sym_simbolos(nombres):
    """Crea los símbolos de una lista de nombres con su nombre LaTeX."""
    return {n: _Symb(n, _LATEX.get(n, n), real=True) for n in nombres}


def modelo(tema):
    """Modelo numérico PÚBLICO del tema (lo usa el notebook del alumno)."""
    return _modelo(tema)


def _modelo(tema):
    """Construye el modelo numérico de un tema: f(x, **params) y metadatos."""
    info = CATALOGO[tema]
    trans = standard_transformations
    locales = {'pi': sp.pi, 'exp': sp.exp, 'log': sp.log, 'log10': lambda z: sp.log(z, 10),
               'sqrt': sp.sqrt, 'cos': sp.cos, 'sin': sp.sin, 'tan': sp.tan,
               'cosh': sp.cosh, 'abs': sp.Abs}
    nombres_params = list(info['params'].keys())
    sym_params = _sym_simbolos(nombres_params)
    if info.get('sistema'):
        syms = [_Symb(v, _LATEX.get(v, v), real=True) for v in info['vars']]
        ld = dict(locales, **sym_params)
        ld.update({str(s): s for s in syms})
        exprs = [parse_expr(e, local_dict=ld, transformations=trans) for e in info['exprs']]
        F = sp.lambdify([syms, [sym_params[p] for p in nombres_params]], exprs, 'numpy')

        def f_vec(X, vals):
            return np.array(F(list(X), [vals[p] for p in nombres_params]), dtype=float)
        return {'info': info, 'params': nombres_params, 'vars': info['vars'],
                'F': f_vec, 'sym': (syms, exprs, sym_params), 'sistema': True}

    x = _Symb(_VAR_REAL[tema], _LATEX.get(_VAR_REAL[tema], _VAR_REAL[tema]), real=True)
    ld = dict(locales, **sym_params)
    ld[str(x)] = x
    expr = parse_expr(info['expr'], local_dict=ld, transformations=trans)
    f = sp.lambdify([x, [sym_params[p] for p in nombres_params]], expr, 'numpy')

    estim = None
    if info.get('estimacion'):
        _e = parse_expr(info['estimacion'], local_dict=dict(locales, **sym_params), transformations=trans)
        _ef = sp.lambdify([[sym_params[p] for p in nombres_params]], _e, 'numpy')

        def estim(vals):
            return float(_ef([vals[p] for p in nombres_params]))

    def f_scalar(xv, vals):
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:
                return float(f(xv, [vals[p] for p in nombres_params]))
            except TypeError:
                return float('nan')

    def f_arr(xs, vals):
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            return np.asarray(f(xs, [vals[p] for p in nombres_params]), dtype=float)

    # Coeficientes del polinomio (de mayor a menor grado) si el caso es
    # polinómico: los necesita Bairstow, que solo recibe `f` por el contrato.
    # Se obtienen simbólicamente, así que siguen la incertidumbre de la muestra.
    coefs_fn = None
    if info.get('polinomico'):
        try:
            _cexprs = sp.Poly(expr, x).all_coeffs()
            _cf = sp.lambdify([[sym_params[p] for p in nombres_params]], _cexprs, 'numpy')

            def coefs_fn(vals):
                return [float(c) for c in _cf([vals[p] for p in nombres_params])]
        except Exception:
            coefs_fn = None

    return {'info': info, 'params': nombres_params, 'vars': [str(x)],
            'f': f_scalar, 'f_arr': f_arr, 'sym': (x, expr, sym_params),
            'sistema': False, 'estimacion': estim, 'coefs': coefs_fn}
