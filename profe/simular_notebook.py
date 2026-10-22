# -*- coding: utf-8 -*-
"""
profe/simular_notebook.py — Ejecuta el cuaderno del examen sin abrir Colab.

    python profe/simular_notebook.py --experto
    python profe/simular_notebook.py --experto --ofuscado

Recorre las celdas de código en un mismo espacio de nombres y después hace lo
que haría un alumno perfecto: escribe `raiz(f, a, b)`, contesta los widgets
teóricos con la opción correcta y llama a `calificar(debug=True)` (sin POST
real). Comprueba que un alumno perfecto saca 100 / 100.

Con `--ofuscado` fuerza el cuaderno a cargar `grader_ofuscado.txt`.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('PLOTLY_RENDERER', 'json')
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

CUADERNO = os.path.join(RAIZ, 'U2_Examen.ipynb')
MAGIC = re.compile(r'^\s*[%!][A-Za-z]')
NC_EJEMPLO = 'm16330887@hermosillo.tecnm.mx'


class MarcoFalso(object):
    def __init__(self, globs):
        self.f_globals = globs


def raiz_experto(f, a, b, tol=1e-6, max_iter=100):
    """El método que escribiría un alumno que domina el tema: raíz en [a, b].

    Busca el primer cambio de signo en [a, b] (igual que el oráculo) y lo
    afina con brentq. Devuelve (raiz, iteraciones).
    """
    import numpy as np
    from scipy.optimize import brentq
    import math
    xs = np.linspace(a, b, 500)
    ys = np.asarray([f(v) for v in xs], dtype=float)
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        ya, yb = ys[i], ys[i + 1]
        if not (np.isfinite(ya) and np.isfinite(yb)):
            continue
        if ya == 0:
            return float(xa), 1
        if ya * yb < 0:
            r = brentq(f, float(xa), float(xb), xtol=1e-12, maxiter=200)
            it = max(1, int(math.ceil(math.log2((b - a) / tol)))) if tol > 0 else 1
            return float(r), it
    return float('nan'), max_iter


def celdas_de_codigo(ruta):
    with io.open(ruta, encoding='utf-8') as f:
        nb = json.load(f)
    salida = []
    for celda in nb.get('cells', []):
        if celda.get('cell_type') != 'code':
            continue
        fuente = celda.get('source', '')
        if isinstance(fuente, list):
            fuente = ''.join(fuente)
        salida.append('\n'.join(l for l in fuente.splitlines() if not MAGIC.match(l)))
    return salida


def ejecutar(ofuscado=False, verboso=False, bundle=False):
    if not os.path.exists(CUADERNO):
        raise SystemExit('No encuentro %s' % CUADERNO)
    celdas = celdas_de_codigo(CUADERNO)
    forzado = 'blob' if ofuscado else ('bundle' if bundle else None)
    print('Cuaderno : %s  (%d celdas de código)'
          % (os.path.basename(CUADERNO), len(celdas)))
    print('Motor    : %s' % (forzado or 'el que diga el cuaderno (MODO_MOTOR)'))

    patron_modo = r'MODO_MOTOR\s*=\s*["\'][a-z]+["\']'
    if forzado and not any(re.search(patron_modo, c) for c in celdas):
        raise SystemExit('El cuaderno no declara MODO_MOTOR.')

    ns = {'__name__': '__main__', 'alumno_id': NC_EJEMPLO}
    fallos = []
    for i, codigo in enumerate(celdas, start=1):
        if forzado:
            codigo = re.sub(patron_modo, 'MODO_MOTOR = "%s"' % forzado, codigo)
        if re.search(r'(?m)^\s*calificar\(', codigo) and not os.environ.get('MN_ENVIAR_REAL'):
            print('   [celda %d] se omite calificar() (haría POST real)' % i)
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(codigo, '<celda %d>' % i, 'exec'), ns)
        except BaseException as exc:                       # noqa: BLE001
            fallos.append((i, exc, buf.getvalue().strip().splitlines()[-3:]))
        if verboso:
            print('   celda %2d %s' % (i, 'FALLA' if any(f[0] == i for f in fallos) else 'OK'))

    if fallos:
        print('\n❌ Celdas que fallaron:')
        for i, exc, cola in fallos:
            print('   celda %d: %r' % (i, exc))
            for linea in cola:
                print('       %s' % linea)
        return 1
    print('   todas las celdas corrieron sin excepciones')

    ex = ns.get('MI_TAREA')
    if ex is None:
        print('❌ El cuaderno no creó MI_TAREA.')
        return 1

    ns['raiz'] = raiz_experto
    marco = MarcoFalso(ns)
    for i, p in enumerate(ex.preguntas, start=1):
        if p['tipo'] == 'caso':
            continue
        w = ns.get('resp_%d' % i)
        if w is None:
            continue
        if p['tipo'] == 'teorica':
            w.value = int(p['correcta'])
        elif p['tipo'] == 'raiz':
            w.value = float(p['ref'])

    res = ns['calificar'](debug=True, marco=marco)
    calif = res.get('calificacion', 0.0)
    maximo = res.get('maximo', 0)
    print('\nAlumno perfecto: %.2f / %d  (mínimo para enviar %g)' % (calif, maximo, res.get('minimo', 0)))

    problemas = []
    if abs(calif - float(maximo)) > 1e-6:
        problemas.append('un alumno perfecto no saca el máximo (saca %.2f)' % calif)

    print()
    if problemas:
        for p in problemas:
            print('❌ %s' % p)
        return 1
    print('=' * 74)
    print(' RESULTADO: TODO OK')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Ejecuta el cuaderno como si fuera el alumno.')
    ap.add_argument('--experto', action='store_true')
    ap.add_argument('--ofuscado', action='store_true')
    ap.add_argument('--bundle', action='store_true')
    ap.add_argument('--verboso', action='store_true')
    args = ap.parse_args()
    sys.exit(ejecutar(ofuscado=args.ofuscado, verboso=args.verboso, bundle=args.bundle))
