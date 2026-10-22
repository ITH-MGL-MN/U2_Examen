# -*- coding: utf-8 -*-
"""
profe/core/montecarlo.py — Simulación de Monte Carlo.

Perturba los parámetros del caso y corre cada método del equipo sobre las
mismas N muestras, comparando contra el oráculo muestra por muestra.
"""
import numpy as np

from .helpers import _bi
from .catalogo import (CATALOGO, _norm, familia, MAX_ITER, TAU_X, TAU_ABS, TAU_F, UMBRAL_EXITO)
from .metodos import SinCambioDeSigno, Divergente, NoAplica, Estancado
from .modelos import _modelo
from .oraculo import oraculo_escalar_todas, oraculo_sistema


# Texto legible de cada motivo de fallo (lo usan el cuaderno y los consejos).
MOTIVOS_TEXTO = {
    'ok': 'acertó la raíz',
    'sin_cambio_signo': 'no arrancó: el intervalo no encierra la raíz',
    'divergente': 'divergió (la iteración se fue a infinito)',
    'estancado': 'se estancó (derivada o denominador nulo)',
    'no_converge': 'no convergió (agotó las iteraciones)',
    'otra_raiz': 'convergió a OTRA raíz (no la física)',
    'complejo': 'se fue al plano complejo',
    'no_finito': 'devolvió un valor no finito',
    'impreciso': 'se detuvo fuera de la tolerancia',
    'no_aplica': 'no aplica a este caso',
    'error': 'error inesperado',
    'sin_raiz_ref': 'el oráculo no halló raíz en esa muestra',
}


def _motivo_de_fallo(exc, raiz_i, it, resid, max_iter, sistema=False):
    """Traduce un fallo en un motivo concreto, para no llamar 'divergencia' a todo."""
    if exc is not None:
        return exc
    if not np.all(np.isfinite(raiz_i)):
        return 'no_finito' if sistema else 'complejo'
    if resid is not None and np.isfinite(resid) and resid <= TAU_F:
        # El residuo es ~0: sí llegó a UNA raíz, pero no a la que buscábamos.
        return 'otra_raiz'
    if it >= max_iter:
        return 'no_converge'
    return 'impreciso'


def muestrear(tema, N=1000, seed=0):
    """Perturba los parámetros del tema (normal o uniforme). Misma semilla -> comparación justa."""
    info = CATALOGO[tema]
    rng = np.random.default_rng(int(seed) % 2 ** 32)
    M = {}
    for p, spec in info['params'].items():
        if spec[0] == 'u':
            M[p] = rng.uniform(spec[1], spec[2], size=N)
        else:
            mu, sig = float(spec[1]), float(spec[2])
            M[p] = mu * (1 + sig * rng.normal(size=N)) if sig else np.full(N, mu)
    return M


def _a_real(r):
    """Resultado escalar -> float real, o nan si salió complejo (Müller/Bairstow
    se fueron del eje real: es una muestra fallida, no la raíz física)."""
    try:
        z = complex(r)
    except (TypeError, ValueError):
        return float('nan')
    if abs(z.imag) <= 1e-9 * max(1.0, abs(z.real)):
        return float(z.real)
    return float('nan')


def _a_vector_real(r, n):
    """Resultado de sistema -> arreglo float de tamaño n, o nan si no encaja."""
    try:
        v = np.asarray(r, dtype=float)
    except (TypeError, ValueError):
        return np.full(n, np.nan)
    return v if v.shape == (n,) else np.full(n, np.nan)


def _n_scan_oraculo(defecto=500):
    """Puntos del escaneo del oráculo (config: monte_carlo.n_scan_oraculo).

    El oráculo se lleva ~84% del tiempo de la simulación, así que bajar el
    escaneo es la única optimización que se nota. Verificado en los 19 temas:
    con 500 puntos la raíz de referencia coincide a 1e-14 con la de 2000.
    """
    try:
        from profe.config import obtener_configuracion
        cfg, _ = obtener_configuracion()
        return int(cfg.get('monte_carlo', {}).get('n_scan_oraculo', defecto))
    except Exception:
        return defecto


def evaluar_tema(tema, metodos, N=1000, seed=0, tol=1e-6, max_iter=MAX_ITER,
                 intervalo=None, arranques=None):
    """Corre el Monte Carlo: por método y por muestra, y compara contra el oráculo.

    metodos   : {nombre: g} donde g(f, p0, p1, tol, max_iter) -> (raiz, n_iter)
                (en sistemas: g(F, x0, x1, tol, max_iter) -> (vector, n_iter))
    intervalo : (opcional) par (a, b) propuesto por el equipo. Si se pasa, los
                métodos que necesitan un PARÉNTESIS (familias 'cerrado' e
                'híbrido') lo usan tal cual en vez del arranque con error de
                aislamiento. Así se ve que, con un intervalo que sí encierra la
                raíz, esos métodos no fallan nunca. Los métodos abiertos y los
                de polinomios conservan su arranque (no usan el intervalo).
    arranques : (opcional) punto(s) de arranque para los métodos que NO usan
                intervalo: (p0, p1) en una ecuación (Newton usa p0; Secante y
                Müller usan los dos; Bairstow los toma como (r0, s0)) o el vector
                x0 en un sistema. A diferencia del arranque simulado, este se usa
                IGUAL en las N muestras.

    Devuelve {'metodos': {nombre: métricas}, 'raiz_ref': ..., 'intervalo': ...,
              'arranques': ...}.
    """
    mod = _modelo(tema)
    info = mod['info']
    if arranques is not None:
        _arr = np.asarray(arranques, dtype=float).ravel()
        _n_esp = len(info['x0']) if mod['sistema'] else 2
        if _arr.size != _n_esp or not np.all(np.isfinite(_arr)):
            raise ValueError(
                'arranques debe ser %d valor(es) finito(s)%s; recibí %r'
                % (_n_esp, ' (las incógnitas del sistema)' if mod['sistema'] else ' (p0, p1)',
                   arranques))
        arranques = tuple(float(v) for v in _arr)
    M = muestrear(tema, N=N, seed=seed)
    nombres = [_norm(k) for k in metodos.keys()]
    n_scan = _n_scan_oraculo()

    # ---- Oráculo por muestra (PRIMERO: define la verdad de referencia) ----
    raiz_ref = np.full((N,) if not mod['sistema'] else (N, len(info['x0'])), np.nan)
    for i in range(N):
        vals = {p: M[p][i] for p in mod['params']}
        if mod['sistema']:
            R, ok = oraculo_sistema(lambda X: mod['F'](X, vals), info['x0'])
            raiz_ref[i] = R if ok else np.nan
        else:
            _lo, _hi = info['intervalo']
            _rts = oraculo_escalar_todas(lambda xs_: mod['f_arr'](xs_, vals), _lo, _hi, n_scan=n_scan)
            if not _rts:
                raiz_ref[i] = np.nan
            elif mod.get('estimacion') is not None:
                raiz_ref[i] = _bi.min(_rts, key=lambda _r: abs(_r - mod['estimacion'](vals)))
            else:
                raiz_ref[i] = _rts[0]

    # ---- Arranques por muestra (modela el error de aislamiento del alumno) ----
    rng = np.random.default_rng((int(seed) + 7919) % 2 ** 32)
    if mod['sistema']:
        base = np.array(info['x0'], dtype=float)
        _arr = base[None, :] * (1 + 0.35 * rng.normal(size=(N, base.size)))
        lejos = rng.random(N) < 0.15
        if lejos.any():
            _arr[lejos] = base[None, :] * (1 + 0.9 * rng.normal(size=(int(lejos.sum()), base.size)))
        p0 = p1 = None
    else:
        lo, hi = info['intervalo']
        anch = float(hi - lo)
        r0 = np.where(np.isfinite(raiz_ref), raiz_ref, 0.5 * (lo + hi))
        bien = rng.random(N) >= 0.15
        mag = np.where(bien, rng.uniform(0.00, 0.15, N),
                       rng.uniform(0.30, 1.20, N)) * anch
        signo = np.where(rng.random(N) < 0.5, -1.0, 1.0)
        x_est = np.clip(r0 + signo * mag, lo, hi)
        w = 0.25 * anch
        hacia = np.where(r0 <= x_est, -1.0, 1.0)
        p0 = x_est
        p1 = np.clip(x_est + hacia * w, lo, hi)

    # ---- Cada método ----
    salida = {}
    for nombre, g in metodos.items():
        can = _norm(nombre)
        # Con un intervalo propuesto, solo lo usan los métodos que necesitan un
        # paréntesis (cerrado/híbrido). Los abiertos y los de polinomios siguen
        # con su arranque: Bairstow, de hecho, espera (r0, s0) y NO un intervalo.
        usa_intervalo = (intervalo is not None) and (familia(can) in ('cerrado', 'hibrido'))
        # Y al revés: solo los métodos que NO usan intervalo (abiertos y
        # polinómicos) aceptan el punto de arranque que proponga el equipo. Así
        # los dos controles son independientes y se leen igual que en el cuaderno.
        usa_arranque = (arranques is not None) and (familia(can) not in ('cerrado', 'hibrido'))
        n_iter = np.full(N, np.nan)
        raiz = np.full(N, np.nan) if not mod['sistema'] else np.full((N, len(info['x0'])), np.nan)
        exito = np.zeros(N, dtype=bool)
        motivo = np.array([''] * N, dtype=object)
        for i in range(N):
            vals = {p: M[p][i] for p in mod['params']}
            exc = None
            try:
                if mod['sistema']:
                    _x0 = np.array(arranques, dtype=float) if usa_arranque else np.array(_arr[i])
                    res = g(lambda X: mod['F'](X, vals), _x0, None, tol, max_iter)
                else:
                    _fx = (lambda xv: mod['f'](xv, vals))
                    # Los casos polinómicos exponen sus coeficientes colgándolos
                    # de la función: el contrato solo pasa `f`, y Bairstow los
                    # necesita para arrancar.
                    if mod.get('coefs') is not None:
                        try:
                            _fx.coefs = mod['coefs'](vals)
                        except Exception:
                            pass
                    if usa_intervalo:
                        _a, _b = float(intervalo[0]), float(intervalo[1])
                    elif usa_arranque:
                        _a, _b = float(arranques[0]), float(arranques[1])
                    else:
                        _a, _b = float(p0[i]), float(p1[i])
                    res = g(_fx, _a, _b, tol, max_iter)
                if isinstance(res, tuple):
                    r, it = res[0], res[1]
                else:
                    r, it = res, np.nan
                it = int(it) if it is not None and np.isfinite(it) else max_iter
            except SinCambioDeSigno:
                exc, r, it = 'sin_cambio_signo', np.nan, max_iter
            except NoAplica:
                exc, r, it = 'no_aplica', np.nan, max_iter
            except (Estancado, ZeroDivisionError):
                exc, r, it = 'estancado', np.nan, max_iter
            except (Divergente, OverflowError):
                # OverflowError sale de casos como d**2.5 con una d enorme: la
                # iteración se disparó, que es una divergencia y no un fallo raro.
                exc, r, it = 'divergente', np.nan, max_iter
            except Exception:
                exc, r, it = 'error', np.nan, max_iter
            if mod['sistema'] and np.isscalar(r):
                # En los sistemas el resultado es un vector: el fallo (un escalar
                # nan) se expande para que la clasificación lo vea no finito.
                r = np.full(len(info['x0']), np.nan)
            n_iter[i] = it
            raiz[i] = _a_vector_real(r, len(info['x0'])) if mod['sistema'] else _a_real(r)
            ref = raiz_ref[i]
            if not np.all(np.isfinite(ref)):
                motivo[i] = 'sin_raiz_ref'
                continue
            escala = float(np.max(np.abs(ref)))
            tol_ok = _bi.max(TAU_X * escala, TAU_ABS)
            resid = None
            if mod['sistema']:
                if np.all(np.isfinite(raiz[i])):
                    dif = float(np.max(np.abs(raiz[i] - ref)))
                    resid = float(np.max(np.abs(mod['F'](np.asarray(raiz[i]), vals))))
                else:
                    dif = np.inf
            else:
                if np.isfinite(raiz[i]):
                    dif = _bi.abs(raiz[i] - ref)
                    resid = _bi.abs(mod['f'](float(raiz[i]), vals))
                else:
                    dif = np.inf
            exito[i] = bool(dif <= tol_ok and it <= max_iter)
            if exito[i]:
                motivo[i] = 'ok'
            else:
                motivo[i] = _motivo_de_fallo(exc, raiz[i], it, resid, max_iter, mod['sistema'])
        err_ref = np.array([
            (np.max(np.abs(np.asarray(raiz[i]) - raiz_ref[i]) / np.maximum(np.abs(raiz_ref[i]), 1e-12))
             if (np.all(np.isfinite(raiz_ref[i])) and np.all(np.isfinite(raiz[i]))) else np.nan)
            for i in range(N)]) if mod['sistema'] else np.abs(raiz - raiz_ref) / np.maximum(np.abs(raiz_ref), 1e-12)
        salida[can] = {
            'n_iter': n_iter,
            'raiz': raiz,
            'exito': exito,
            'pct_exito': float(np.mean(exito)) if N else 0.0,
            'diverge': int(np.sum(~exito)),
            'pct_diverge': float(np.mean(~exito)) if N else 1.0,
            'err_ref': err_ref,
            'familia': familia(can),
            'motivo': motivo,
            'motivos': {k: int(np.sum(motivo == k)) for k in sorted(set(motivo)) if k},
            'usa_intervalo': bool(usa_intervalo),
            'usa_arranque': bool(usa_arranque),
            'sospechoso': bool(np.nanmedian(n_iter) < 2),
            'cuenta': bool(np.mean(exito) >= UMBRAL_EXITO and np.nanmedian(n_iter) >= 2),
        }
    return {'tema': tema, 'titulo': info['titulo'], 'N': N, 'seed': seed,
            'raiz_ref': raiz_ref, 'metodos': salida, 'sistema': mod['sistema'],
            'polinomico': bool(info.get('polinomico')), 'intervalo': intervalo,
            'arranques': arranques}
