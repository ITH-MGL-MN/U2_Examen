# -*- coding: utf-8 -*-
"""
profe/core/metodos.py — Métodos numéricos YA IMPLEMENTADOS (los pone el motor).

El alumno NO programa los métodos: esta tarea es teórica. El cuaderno solo
muestra un selector y el alumno elige qué métodos comparar; las
implementaciones viven aquí, dentro del motor, así que quedan **ocultas**.

`mapa_metodos(es_sistema)` devuelve el diccionario {nombre: g} con el adaptador
`g(f, p0, p1, tol, max_iter) -> (raiz, n_iter)` (en sistemas, `g(F, x0, _x1, tol,
max_iter) -> (vector_raiz, n_iter)`), que es el contrato que ya consume
`montecarlo.evaluar_tema`.

Prueba local:  python -m profe.core.metodos
"""
import numpy as np


# ===========================================================================
# 0) MOTIVOS DE FALLO — tipos propios, para que la simulación pueda distinguir
#    POR QUÉ falló un método y no meterlo todo en el mismo saco.
# ===========================================================================
class SinCambioDeSigno(ValueError):
    """El intervalo no encierra la raíz: el método no puede ni arrancar.

    No es una divergencia del método: es que el paréntesis (el aislamiento)
    está mal. Lo lanzan bisección, falsa posición, punto fijo modificado y
    Brent, que necesitan f(a)·f(b) < 0.
    """


class Divergente(RuntimeError):
    """La iteración se fue a un valor no finito: el despeje no es contractivo."""


class NoAplica(ValueError):
    """El método no se puede aplicar a este caso (p. ej. Bairstow sin coeficientes)."""


class Estancado(RuntimeError):
    """La iteración no puede seguir: derivada ~0, denominador nulo o Jacobiano singular."""


# ===========================================================================
# 1) ECUACIÓN DE PRUEBA (para el banco de pruebas local)
#    f(x) = x^2 - 2  ->  raíz = sqrt(2) = 1.41421356...
# ===========================================================================
def f_prueba(x):
    return x ** 2 - 2.0


# ===========================================================================
# 2) MÉTODOS CERRADOS (intervalo)
# ===========================================================================
def biseccion(f, p0, p1, tol=1e-6, max_iter=100):
    """Bisección. Requiere f(a)*f(b) < 0."""
    a, b = min(p0, p1), max(p0, p1)
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa * fb > 0:
        raise SinCambioDeSigno('La bisección necesita cambio de signo en [a, b].')
    for k in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0 or (b - a) / 2 < tol:
            return m, k
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b), max_iter


def falsa_posicion(f, p0, p1, tol=1e-6, max_iter=100):
    """Falsa posición (regula falsi)."""
    a, b = min(p0, p1), max(p0, p1)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise SinCambioDeSigno('La falsa posición necesita cambio de signo en [a, b].')
    x_prev = None
    for k in range(1, max_iter + 1):
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)
        if fx == 0 or (x_prev is not None and abs(x - x_prev) < tol):
            return x, k
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_prev = x
    return x, max_iter


def punto_fijo_modificado(f, p0, p1, tol=1e-6, max_iter=100):
    """Punto fijo modificado (falsa posición modificada).

    Variante de la falsa posición que evita el "estancamiento" de un extremo:
    cuando un extremo queda fijo dos iteraciones seguidas, su valor de función
    se divide a la mitad para que el intervalo sí se cierre (Chapra, §5.4).
    """
    a, b = min(p0, p1), max(p0, p1)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise SinCambioDeSigno('Necesita cambio de signo en [a, b].')
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    est_a = est_b = 0   # iteraciones seguidas que 'a' / 'b' quedan fijos
    x_prev = None
    for k in range(1, max_iter + 1):
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)
        if fx == 0 or (x_prev is not None and abs(x - x_prev) < tol):
            return x, k
        if fa * fx < 0:
            # raíz en [a, x]: 'b' se mueve y 'a' queda fijo -> suelta 'a'.
            b, fb = x, fx
            est_a += 1
            est_b = 0
            if est_a >= 2:
                fa = 0.5 * fa
        else:
            # raíz en [x, b]: 'a' se mueve y 'b' queda fijo -> suelta 'b'.
            a, fa = x, fx
            est_b += 1
            est_a = 0
            if est_b >= 2:
                fb = 0.5 * fb
        x_prev = x
    return x, max_iter


# ===========================================================================
# 3) MÉTODOS ABIERTOS
# ===========================================================================
def punto_fijo(f, p0, _p1=None, tol=1e-6, max_iter=100, g=None):
    """Iteración de punto fijo: x_{k+1} = g(x_k).

    `f` se usa para comprobar; `g` es el despeje. Si no se pasa `g`, se usa
    x -> x - f(x) (relajación débil).
    """
    gg = g if g is not None else (lambda x: x - f(x))
    x = float(p0)
    for k in range(1, max_iter + 1):
        x_new = gg(x)
        if not np.isfinite(x_new):
            raise Divergente('Punto fijo divergió (valor no finito).')
        if abs(x_new - x) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


def newton_raphson(f, p0, _p1=None, tol=1e-6, max_iter=100, df=None):
    """Newton-Raphson. `df` es la derivada; si no se pasa, se aproxima numéricamente."""
    if df is None:
        def df(x, h=1e-7):
            return (f(x + h) - f(x - h)) / (2 * h)
    x = float(p0)
    for k in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if abs(dfx) < 1e-14:
            raise Estancado('Derivada casi nula: Newton-Raphson no puede continuar.')
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


def nr_modificado(f, p0, _p1=None, tol=1e-6, max_iter=100, df=None, d2f=None):
    """Newton-Raphson modificado (para raíces múltiples).

    x_{k+1} = x_k - f f' / ((f')^2 - f f'').  Derivadas numéricas por defecto.
    """
    if df is None:
        def df(x, h=1e-6):
            return (f(x + h) - f(x - h)) / (2 * h)
    if d2f is None:
        def d2f(x, h=1e-5):
            return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    x = float(p0)
    for k in range(1, max_iter + 1):
        fx, d1, d2 = f(x), df(x), d2f(x)
        den = d1 * d1 - fx * d2
        if abs(den) < 1e-14:
            raise Estancado('Denominador casi nulo en NR modificado.')
        x_new = x - fx * d1 / den
        if abs(x_new - x) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


def secante(f, p0, p1, tol=1e-6, max_iter=100):
    """Secante. No requiere cambio de signo ni derivada."""
    x0, x1 = float(p0), float(p1)
    f0, f1 = f(x0), f(x1)
    for k in range(1, max_iter + 1):
        if abs(f1 - f0) < 1e-14:
            raise Estancado('División por cero en la secante.')
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2, k
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    return x1, max_iter


def secante_modificada(f, p0, _p1=None, tol=1e-6, max_iter=100, delta=1e-6):
    """Secante modificada (perturbación delta en vez de dos puntos)."""
    x = float(p0)
    for k in range(1, max_iter + 1):
        fx = f(x)
        fdx = f(x + delta * x)
        den = fdx - fx
        if abs(den) < 1e-14:
            raise Estancado('División por cero en la secante modificada.')
        x_new = x - delta * x * fx / den
        if abs(x_new - x) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


# ===========================================================================
# 4) MÉTODO HÍBRIDO
# ===========================================================================
def brent(f, p0, p1, tol=1e-6, max_iter=100):
    """Método de Brent (bisección + interpolación cuadrática inversa)."""
    from scipy.optimize import brentq
    a, b = min(p0, p1), max(p0, p1)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise SinCambioDeSigno('Brent necesita cambio de signo en [a, b].')
    raiz, r = brentq(f, a, b, xtol=tol, rtol=4 * np.finfo(float).eps,
                     maxiter=max_iter, full_output=True)
    n_iter = int(r.iterations) if hasattr(r, 'iterations') else max_iter
    return float(raiz), n_iter


# ===========================================================================
# 5) MÉTODOS PARA POLINOMIOS
# ===========================================================================
def _real(z):
    """Un complejo a float si su parte imaginaria es redondeo (|Im| ~ 0)."""
    z = complex(z)
    if abs(z.imag) <= 1e-12 * max(1.0, abs(z.real)):
        return float(z.real)
    return z


def muller(f, p0, p1, p2=None, tol=1e-6, max_iter=100):
    """Método de Müller (parábola por tres puntos).

    Acepta (f, x0, x1, x2) o, si solo se pasan p0 y p1, arma un tercer punto.
    """
    import cmath
    x0, x1 = float(p0), float(p1)
    x2 = float(p2) if p2 is not None else 0.5 * (x0 + x1)
    for it in range(1, max_iter + 1):
        f0, f1, f2 = complex(f(x0)), complex(f(x1)), complex(f(x2))
        h0, h1 = x1 - x0, x2 - x1
        d0 = (f1 - f0) / h0
        d1 = (f2 - f1) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f2
        disc = cmath.sqrt(b * b - 4 * a * c)
        den = b + disc if abs(b + disc) > abs(b - disc) else b - disc
        x3 = x2 - 2 * c / den
        # Se devuelve a real cuando sigue sobre el eje real: si no, el tipo
        # complejo se propaga y `f` (que solo acepta reales) devuelve nan.
        x3 = _real(x3)
        if abs(f(x3)) < tol or abs(x3 - x2) < tol * max(1.0, abs(x3)):
            return _real(x3), it
        x0, x1, x2 = x1, x2, x3
    return _real(x2), max_iter


def bairstow(f, p0, p1, tol=1e-6, max_iter=100, coefs=None):
    """Método de Bairstow (extrae factores cuadráticos x^2 - r x - s).

    Necesita los coeficientes del polinomio (de mayor a menor grado) en `coefs`;
    sin ellos no puede empezar, así que levanta excepción. Devuelve TODAS las
    raíces (reales y complejas) de la lista.
    """
    import cmath
    if coefs is None:
        if hasattr(f, 'coefs'):
            coefs = f.coefs
        else:
            raise NoAplica('Bairstow necesita los coeficientes (pasa coefs=[...]).')
    a = [float(c) for c in coefs]
    r, s = float(p0), float(p1)
    raices, n_iter = [], 0

    def _paso(a, r, s):
        n = len(a) - 1
        b = [0.0] * (n + 1)
        b[0] = a[0]
        if n >= 1:
            b[1] = a[1] + r * b[0]
        for k in range(2, n + 1):
            b[k] = a[k] + r * b[k - 1] + s * b[k - 2]
        c = [0.0] * (n + 1)
        c[0] = b[0]
        if n >= 1:
            c[1] = b[1] + r * c[0]
        for k in range(2, n + 1):
            c[k] = b[k] + r * c[k - 1] + s * c[k - 2]
        b0, b1 = b[n], b[n - 1]
        c1, c2, c3 = c[n - 1], c[n - 2], c[n - 3]
        det = c2 * c2 - c1 * c3
        if abs(det) < 1e-16:
            raise Estancado('Bairstow: sistema de corrección degenerado.')
        dr = (b0 * c3 - b1 * c2) / det
        ds = (b1 * c1 - b0 * c2) / det
        return b, dr, ds

    def _factor(r, s):
        disc = cmath.sqrt(r * r + 4.0 * s)
        return _real((r + disc) / 2.0), _real((r - disc) / 2.0)

    def _cuadratica(co):
        A, B, C = co[0], co[1], co[2]
        disc = cmath.sqrt(B * B - 4 * A * C)
        return _real((-B + disc) / (2 * A)), _real((-B - disc) / (2 * A))

    while len(a) > 3:
        n = len(a) - 1
        b = None
        for _ in range(int(max_iter)):
            b, dr, ds = _paso(a, r, s)
            r, s = r + dr, s + ds
            n_iter += 1
            if abs(dr) < tol * max(1.0, abs(r)) and abs(ds) < tol * max(1.0, abs(s)):
                break
        raices.extend(_factor(r, s))
        a = b[:n - 1]
        r, s = -0.5, -0.5

    if len(a) == 3:
        raices.extend(_cuadratica(a))
    elif len(a) == 2:
        raices.append(_real(-a[1] / a[0]))
    return raices, n_iter


def _bairstow_raiz(f, p0, p1, tol=1e-6, max_iter=100):
    """Adaptador de Bairstow al contrato (raiz, n_iter).

    Bairstow devuelve TODAS las raíces; para comparar contra el oráculo se toma
    la **raíz real más cercana al arranque** `p0` (el mismo papel que cumplen
    los arranques en los demás métodos). Si no hay ninguna real, devuelve nan.
    """
    raices, n_iter = bairstow(f, p0, p1, tol=tol, max_iter=max_iter)
    reales = []
    for r in raices:
        z = complex(r)
        if abs(z.imag) <= 1e-9 * max(1.0, abs(z.real)):
            reales.append(float(z.real))
    if not reales:
        return float('nan'), n_iter
    return min(reales, key=lambda r: abs(r - float(p0))), n_iter


# ===========================================================================
# 6) SISTEMAS (2x2) — para T14, T15, T19
# ===========================================================================
def newton_sistema(F, x0, _x1=None, tol=1e-6, max_iter=100, J=None):
    """Newton-Raphson multivariable con Jacobiano numérico."""
    X = np.array(x0, dtype=float)
    n = len(X)
    h = 1e-7
    for k in range(1, max_iter + 1):
        Fx = np.array(F(X), dtype=float)
        if np.max(np.abs(Fx)) < tol:
            return X, k
        Jm = np.zeros((n, n))
        for j in range(n):
            dX = np.zeros(n)
            dX[j] = h
            Jm[:, j] = (np.array(F(X + dX)) - np.array(F(X - dX))) / (2 * h)
        if abs(np.linalg.det(Jm)) < 1e-14:
            raise Estancado('Jacobiano casi singular.')
        X = X - np.linalg.solve(Jm, Fx)
    return X, max_iter


def punto_fijo_sistema(F, x0, _x1=None, tol=1e-6, max_iter=100, G=None):
    """Punto fijo multivariable: X_{k+1} = G(X_k).

    `G` es el despeje (callable X -> np.array). Si no se pasa, se usa
    X -> X - F(X) (relajación débil).
    """
    GG = G if G is not None else (lambda X: np.array(X) - np.array(F(X)))
    X = np.array(x0, dtype=float)
    for k in range(1, max_iter + 1):
        X_new = np.array(GG(X), dtype=float)
        if not np.all(np.isfinite(X_new)):
            raise Divergente('Punto fijo (sistema) divergió.')
        if np.max(np.abs(X_new - X)) < tol:
            return X_new, k
        X = X_new
    return X, max_iter


# ===========================================================================
# 7) MAPA DE MÉTODOS (lo que consume el cuaderno)
# ===========================================================================
def mapa_metodos(es_sistema=False):
    """{nombre: g} con los adaptadores al contrato de `evaluar_tema`.

    Una ecuación :  g(f, p0, p1, tol, max_iter) -> (raiz, n_iter)
    Sistemas     :  g(F, x0, _x1, tol, max_iter) -> (vector_raiz, n_iter)

    Los nombres son los canónicos del catálogo (los mismos que aparecen en
    `metodos_sugeridos`), así que la selección del alumno y la autocalificación
    se comparan sin traducciones.
    """
    if es_sistema:
        return {
            'Newton-Raphson multivariable':
                lambda F, x0, x1, tol, it: newton_sistema(F, x0, tol=tol, max_iter=it),
            'Punto fijo modificado':
                lambda F, x0, x1, tol, it: punto_fijo_sistema(F, x0, tol=tol, max_iter=it),
        }
    return {
        'Bisección':             lambda f, p0, p1, tol, it: biseccion(f, p0, p1, tol, it),
        'Falsa posición':        lambda f, p0, p1, tol, it: falsa_posicion(f, p0, p1, tol, it),
        'Punto fijo modificado': lambda f, p0, p1, tol, it: punto_fijo_modificado(f, p0, p1, tol, it),
        'Punto fijo':            lambda f, p0, p1, tol, it: punto_fijo(f, p0, tol=tol, max_iter=it),
        'Newton-Raphson':        lambda f, p0, p1, tol, it: newton_raphson(f, p0, tol=tol, max_iter=it),
        'NR modificado':         lambda f, p0, p1, tol, it: nr_modificado(f, p0, tol=tol, max_iter=it),
        'Secante':               lambda f, p0, p1, tol, it: secante(f, p0, p1, tol, it),
        'Secante modificado':    lambda f, p0, p1, tol, it: secante_modificada(f, p0, tol=tol, max_iter=it),
        'Brent':                 lambda f, p0, p1, tol, it: brent(f, p0, p1, tol, it),
        'Müller':                lambda f, p0, p1, tol, it: muller(f, p0, p1, tol=tol, max_iter=it),
        'Bairstow':              lambda f, p0, p1, tol, it: _bairstow_raiz(f, p0, p1, tol=tol, max_iter=it),
    }


# ===========================================================================
# 8) BANCO DE PRUEBAS (python -m profe.core.metodos)
# ===========================================================================
if __name__ == '__main__':
    esperado = np.sqrt(2)
    pruebas = [
        ('Bisección', biseccion(f_prueba, 0, 2)),
        ('Falsa posición', falsa_posicion(f_prueba, 0, 2)),
        ('Punto fijo mod.', punto_fijo_modificado(f_prueba, 0, 2)),
        ('Secante', secante(f_prueba, 0, 2)),
        ('Secante mod.', secante_modificada(f_prueba, 1.0)),
        ('Newton-Raphson', newton_raphson(f_prueba, 1.0)),
        ('NR modificado', nr_modificado(f_prueba, 1.0)),
        ('Brent', brent(f_prueba, 0, 2)),
        ('Müller', muller(f_prueba, 0.0, 1.0, 2.0)),
    ]
    print('Raíz de x^2-2 = %.10f\n' % esperado)
    for nombre, (raiz, it) in pruebas:
        err = abs(raiz - esperado)
        estado = 'OK ' if err < 1e-5 else 'REVISAR'
        print('%-18s %s  raiz=%.10f  iter=%d  err=%.2e' % (nombre, estado, raiz, it, err))
    print('\nBairstow se prueba con coeficientes:')
    print('   Bairstow (x^2-2) ->', bairstow(None, 1.0, 1.0, coefs=[1, 0, -2]))
