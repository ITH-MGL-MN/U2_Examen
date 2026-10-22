# -*- coding: utf-8 -*-
"""
profe/ui/cuaderno.py — Interfaz pública del EXAMEN de la Unidad 2.

Es la capa de presentación: el cuaderno solo llama a los nombres de `__all__`.
La lógica de casos (catálogo T1–T19), el oráculo y la simulación de Monte Carlo
viven en `profe/core/` (reutilizados de U2_T4); los bancos teóricos están en
`config/preguntas.yaml` y los pesos/tolerancias en `config/config.yaml`.

Estructura del examen (100 puntos):
  - 5 preguntas teóricas (opción múltiple): FLOTANTES, HORNER, ERROR, METODOS,
    CONVERGENCIA (8 puntos cada una).
  - 2 preguntas de CASO: el alumno escribe `raiz(f, a, b)` eligiendo él el
    método (no se le dice cuál). Se le da SU caso (conocido, 30 pts) y OTRO
    distinto (desconocido, 22 pts). Se califica con Monte Carlo + oráculo.
  - 1 pregunta numérica: reportar la raíz de SU caso (8 pts).

Mecánica de examen (heredada de U1_Examen, simplificada): ventana de horario
por servidor, envío automático si la calificación >= 80 %, y SIN reglas de
reinicio de preguntas.
"""
import builtins as _bi
import inspect

import numpy as np
from IPython.display import Markdown, Math, display

from ..config import obtener_configuracion
from ..core.catalogo import CATALOGO, TAU_X, TAU_ABS, MAX_ITER, es_sistema
from ..core.metodos import biseccion
from ..core.modelos import _modelo
from ..core.montecarlo import muestrear, _a_real
from ..core.oraculo import oraculo_escalar_todas
from ..core.seed import extraer_nc, semilla

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
_CFG, _BANCOS = obtener_configuracion()
_ID = _CFG['tarea']['id']
_EVAL = _CFG['evaluacion']
_CASOS = _CFG['casos']['disponibles']
_ESCALARES = [t for t in _CASOS if not es_sistema(t)]
_RAIZ_CFG = _CFG['raiz']

MIN_APROBACION = float(_EVAL.get('min_aprobacion', 0.80))
TOL_SIMPLE = float(_EVAL.get('tolerancia_simple', 1e-4))
TOL_FUNCION = float(_EVAL.get('tolerancia_funcion', 1e-4))
MAX_ITER_EXAMEN = int(_EVAL.get('max_iter', MAX_ITER))
WEBHOOK_TOKEN = _EVAL.get('webhook_token', '')
APPS_SCRIPT_URL = _EVAL.get('apps_script_url', '')
EXAMEN_TOKEN = _EVAL.get('examen_token', '')
DEBUG_NC = str(_CFG.get('debug', {}).get('nc', ''))


# ---------------------------------------------------------------------------
# Asignación determinista de casos por número de control
# ---------------------------------------------------------------------------
def _caso_de(nc, rol):
    """Caso determinista según el NC y el rol ('conocido'/'desconocido').

    El 'desconocido' es SIEMPRE escalar y distinto del conocido: así se
    garantiza que todo mundo programe el método al menos una vez, aunque su
    caso conocido fuera de sistema.
    """
    if rol == 'conocido':
        pool = _CASOS
        return pool[semilla(_ID, nc) % len(pool)]
    conocido = _caso_de(nc, 'conocido')
    pool = [t for t in _ESCALARES if t != conocido]
    return pool[semilla(_ID, nc, 'otro') % len(pool)]


_CACHE_PARAMS = {}


def _sortear_params(tema, nc, base, intentos=60):
    """Sortea los parámetros del caso para ESE alumno y valida que el caso sirva."""
    mod = _modelo(tema)
    info = mod['info']
    sistema = es_sistema(tema)
    for k in range(intentos):
        M = muestrear(tema, N=1, seed=semilla(_ID, nc, tema, 'nominal', k))
        vals = {p: float(M[p][0]) for p in mod['params']}
        if not all(np.isfinite(v) for v in vals.values()):
            continue
        if sistema:
            return vals
        lo, hi = info['intervalo']
        try:
            rts = oraculo_escalar_todas(lambda xs: mod['f_arr'](xs, vals),
                                        lo, hi, n_scan=800)
        except Exception:
            continue
        if rts and np.isfinite(rts[0]) and abs(rts[0]) < 1e6:
            return vals
    return dict(base)


def _params_nominales(tema, nc=None):
    """Parámetros nominales del caso PARA ESE ALUMNO.

    Se sortean con `semilla(id, nc, tema, 'nominal', intento)`: dos alumnos con
    el MISMO tema ven coeficientes distintos (y por tanto otra ecuación, otro
    intervalo de búsqueda efectivo y otra raíz), pero reproducibles.
    """
    if nc is None:
        nc = _EXAMEN.nc if _EXAMEN is not None else DEBUG_NC
    clave = (tema, str(nc))
    if clave in _CACHE_PARAMS:
        return _CACHE_PARAMS[clave]
    base = {}
    for k, spec in CATALOGO[tema]['params'].items():
        base[k] = 0.5 * (spec[1] + spec[2]) if spec[0] == 'u' else float(spec[1])
    out = _sortear_params(tema, nc, base)
    _CACHE_PARAMS[clave] = out
    return out


def _raiz_nominal(tema, nc=None):
    """Raíz de referencia del caso con los parámetros de ese alumno."""
    mod = _modelo(tema)
    info = mod['info']
    lo, hi = info['intervalo']
    vals = _params_nominales(tema, nc)
    rts = oraculo_escalar_todas(lambda xs: mod['f_arr'](xs, vals), lo, hi, n_scan=1000)
    return rts[0] if rts else None


def _ref_iteraciones(tema, nc=None):
    """Iteraciones de referencia (bisección) sobre el caso nominal del alumno."""
    mod = _modelo(tema)
    info = mod['info']
    lo, hi = info['intervalo']
    vals = _params_nominales(tema, nc)
    try:
        _, it = biseccion(lambda x: mod['f'](x, vals), lo, hi,
                          tol=1e-6, max_iter=MAX_ITER_EXAMEN)
        return max(1, int(it))
    except Exception:
        return 30


# ---------------------------------------------------------------------------
# El objeto del examen
# ---------------------------------------------------------------------------
class Examen:
    def __init__(self, nc, rng):
        self.nc = nc
        self.rng = rng
        self.preguntas = []
        self.pesos = []
        self.temas = {'conocido': _caso_de(nc, 'conocido'),
                      'desconocido': _caso_de(nc, 'desconocido')}
        self.maximo = 0

    def generar(self):
        for slot in _CFG['slots']:
            peso = int(slot['peso'])
            self.maximo += peso
            self.pesos.append(peso)
            tipo = slot['tipo']
            if tipo == 'teorica':
                self.preguntas.append(_teorica_variante(self.rng, slot['banco']))
            elif tipo == 'caso':
                tema = self.temas[slot['rol']]
                if es_sistema(tema):
                    p = _teorica_variante(self.rng, 'SISTEMAS')
                    p['tema'] = tema
                    p['sistema'] = True
                    self.preguntas.append(p)
                else:
                    self.preguntas.append({'tipo': 'caso', 'rol': slot['rol'], 'tema': tema})
            elif tipo == 'raiz':
                tema = self.temas['conocido']
                if es_sistema(tema):
                    p = _teorica_variante(self.rng, 'SISTEMAS')
                    p['tema'] = tema
                    p['sistema'] = True
                    self.preguntas.append(p)
                else:
                    self.preguntas.append({'tipo': 'raiz', 'tema': tema,
                                           'ref': _raiz_nominal(tema, self.nc)})
        return self


# ---------------------------------------------------------------------------
# Render de cada pregunta + widget de respuesta
# ---------------------------------------------------------------------------
def _teorica_variante(rng, nombre_banco):
    banco = _BANCOS[nombre_banco]
    idx = int(rng.integers(len(banco)))
    v = banco[idx]
    return {'tipo': 'teorica', 'texto': v['texto'], 'opciones': v['opciones'],
            'correcta': int(v['correcta'])}


def _render_caso_info(tema):
    info = CATALOGO[tema]
    display(Markdown('**%s.** Incógnita: $%s$ (en %s).'
                     % (info['titulo'], info['incognita'], info['unidad'])))
    display(Math(info['ecuacion']))
    nominal = _params_nominales(tema)
    filas = '  \n'.join('| $%s$ | $%s$ |' % (k, _fmt(v)) for k, v in nominal.items())
    display(Markdown('| Parámetro | Valor nominal |\n|---|---|\n%s' % filas))
    display(Markdown('*Estos coeficientes son **tuyos**: nadie más tiene los mismos '
                     'números, aunque le toque el mismo caso.*'))
    if info.get('sistema'):
        display(Markdown('Punto inicial: $%s$.'
                         % ', $'.join(_fmt(v) for v in info['x0'])))
    else:
        display(Markdown('Intervalo donde aislar la raíz: $[%s,\\ %s]$.'
                         % (_fmt(info['intervalo'][0]), _fmt(info['intervalo'][1]))))


def _render_pregunta(n, p, peso, total):
    if p['tipo'] == 'teorica':
        display(Markdown('#### Pregunta %d' % n))
        display(Markdown(r'**Valor:** $\frac{%d}{%d}$ puntos' % (peso, total)))
        if p.get('sistema') and p.get('tema'):
            display(Markdown('**Tu caso** es un SISTEMA de ecuaciones: aquí no '
                             'programas el método, se evalúa la teoría.'))
            _render_caso_info(p['tema'])
        display(Markdown(p['texto']))
        if p.get('opciones'):
            _opciones = '  \n'.join('%s) %s' % (chr(ord('a') + i), op)
                                   for i, op in enumerate(p['opciones']))
            display(Markdown(_opciones))
    elif p['tipo'] == 'caso':
        tema = p['tema']
        display(Markdown('#### Pregunta %d · Caso de estudio %s' % (n, tema)))
        display(Markdown(r'**Valor:** $\frac{%d}{%d}$ puntos' % (peso, total)))
        _render_caso_info(tema)
        display(Markdown(
            'Escribe la función `raiz(f, a, b)` que regrese `(raiz, iteraciones)`. '
            '**Tú eliges el método numérico** (no se te dice cuál): puede ser cerrado '
            '(usa $[a,b]$ como intervalo), abierto (usa $a$ y $b$ como arranque, o una '
            'derivada numérica), etc. Se probará contra muchos valores de los parámetros.'))
    elif p['tipo'] == 'raiz':
        display(Markdown('#### Pregunta %d · La raíz de tu caso' % n))
        display(Markdown(r'**Valor:** $\frac{%d}{%d}$ puntos' % (peso, total)))
        display(Markdown(
            'Con tu método, ¿a qué raíz llegaste para **tu** caso (el de la pregunta '
            'anterior)? Escribe el número con 4 decimales.'))


def _fmt(v):
    return '%.6g' % float(v)


def _crear_widget(n, p, marco):
    import ipywidgets as widgets
    if p['tipo'] == 'teorica':
        w = widgets.RadioButtons(
            options=[(chr(ord('a') + idx), idx) for idx in range(len(p['opciones']))],
            value=None, disabled=False)
        marco.f_globals['resp_%d' % n] = w
        display(w)
    elif p['tipo'] == 'raiz':
        w = widgets.FloatText(value=0.0, description='Raíz:')
        marco.f_globals['resp_%d' % n] = w
        display(w)


# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_EXAMEN = None
_BLOQUEADO = None
_SEMILLA = 1             # juego de preguntas TEÓRICAS (crece al reiniciar)
_INTENTO = 0             # intentos usados en esta semilla (0 = ninguno)
_SEMILLA_EXAMEN = 1      # semilla con la que se generó _EXAMEN


def generar_tarea(alumno_id, nombre=None):
    global _EXAMEN, _SEMILLA, _INTENTO, _SEMILLA_EXAMEN
    marco = inspect.currentframe().f_back
    nc = extraer_nc(alumno_id)
    estado = _sincronizar(alumno_id, marco)
    if 'bloqueado' in estado:
        raise RuntimeError('🚫 ' + estado['bloqueado'])
    _SEMILLA = estado['semilla']
    _INTENTO = estado['intento']
    rng = np.random.default_rng(semilla(_ID, nc, _SEMILLA))
    _EXAMEN = Examen(nc, rng).generar()
    _SEMILLA_EXAMEN = _SEMILLA
    display(Markdown('Tu caso de estudio es **%s** (semilla %d, intentos usados %d).'
                     % (_EXAMEN.temas['conocido'], _SEMILLA, _INTENTO)))
    return _EXAMEN


def pregunta(n):
    marco = inspect.currentframe().f_back
    ex = _EXAMEN
    if ex is None:
        print('Primero ejecuta la celda de identificación (generar_tarea).')
        return
    i = n - 1
    p = ex.preguntas[i]
    _render_pregunta(n, p, ex.pesos[i], ex.maximo)
    _crear_widget(n, p, marco)


def mi_caso():
    """Vuelve a mostrar TU caso de estudio (ecuación, parámetros e intervalo)."""
    if _EXAMEN is None:
        print('Primero ejecuta la identificación (generar_tarea).')
        return
    tema = _EXAMEN.temas['conocido']
    display(Markdown('### Tu caso: %s' % tema))
    _render_caso_info(tema)


def graficar_caso(titulo=None):
    """Dibuja f(x) del caso (parámetros nominales) en su intervalo."""
    if _EXAMEN is None:
        print('Primero ejecuta la identificación (generar_tarea).')
        return
    import matplotlib.pyplot as plt
    tema = _EXAMEN.temas['conocido']
    mod = _modelo(tema)
    info = mod['info']
    lo, hi = info['intervalo']
    vals = _params_nominales(tema)
    xs = np.linspace(lo, hi, 400)
    with np.errstate(all='ignore'):
        ys = np.asarray(mod['f_arr'](xs, vals), dtype=float)
    ys = np.where(np.isfinite(ys), ys, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, lw=1.8)
    ax.axhline(0, color='k', lw=0.9)
    r = _raiz_nominal(tema)
    if r is not None:
        ax.plot([r], [0], 'o', color='tab:red')
        ax.annotate('raiz ~ %s' % _fmt(r), (r, 0),
                    textcoords='offset points', xytext=(6, 8), color='tab:red')
    ax.set_xlabel('$%s$' % info['incognita'])
    ax.set_ylabel('$f$')
    ax.set_title(titulo or ('Caso %s - %s' % (tema, info['titulo'])))
    ax.grid(alpha=0.3)
    plt.show()


def probar_metodo(f, a, b):
    """Prueba TU `raiz(f, a, b)` en una función cualquiera (para depurar).

        probar_metodo(lambda x: x**2 - 2, 0, 2)     # raíz esperada: sqrt(2)
    """
    marco = inspect.currentframe().f_back
    fn = marco.f_globals.get('raiz') if marco else None
    if fn is None:
        print('Todavía no defines `raiz(f, a, b)`.')
        return
    rts = oraculo_escalar_todas(f, a, b, n_scan=800)
    if not rts:
        display(Markdown('ℹ️ En $[%s, %s]$ no hay cambio de signo: no hay raíz que '
                         'comparar.' % (_fmt(a), _fmt(b))))
        return
    ref = rts[0]
    try:
        res = fn(f, a, b)
        r, it = res if isinstance(res, tuple) else (res, None)
        r = _a_real(r)
    except Exception as e:
        display(Markdown('❌ Tu `raiz` lanzó un error: `%r`' % (e,)))
        return
    dif = abs(r - ref) if np.isfinite(r) else float('inf')
    ok = np.isfinite(r) and dif <= max(TAU_X * max(1.0, abs(ref)), 1e-9)
    display(Markdown(
        '**Prueba de tu `raiz`**\n\n'
        '| | |\n|---|---|\n'
        '| Raíz de referencia | $%s$ |\n'
        '| Tu raíz | $%s$ |\n'
        '| Error absoluto | $%s$ |\n'
        '| Iteraciones | %s |\n'
        '| ¿Acierta? | %s |'
        % (_fmt(ref), _fmt(r) if np.isfinite(r) else '\\text{nan}',
           _fmt(dif) if np.isfinite(dif) else '\\infty',
           '—' if it is None else str(it),
           '✅ sí' if ok else '❌ no')))


# ---------------------------------------------------------------------------
# Calificación
# ---------------------------------------------------------------------------
class ResultadoCalificacion(dict):
    """Resultado de `calificar()` que NO se vuelca completo en la salida.

    Se comporta como dict (el simulador y el profesor pueden leer
    `r['calificacion']`, `r['respuestas']`, etc.), pero el cuaderno no imprime
    el contenido: lo que ve el alumno es solo el resumen que se dibuja con
    `display()`. Así no se filtra el detalle interno ni el cuerpo enviado.
    """

    def _repr_pretty_(self, p, cycle):   # IPython: no imprimir nada
        pass

    def _repr_markdown_(self):           # rich display: nada
        return ''

    def __repr__(self):                  # Python normal: nada
        return ''

    __str__ = __repr__


def _calificar_teorica(p, marco, i):
    w = marco.f_globals.get('resp_%d' % i) if marco else None
    val = w.value if w is not None else None
    return (val is not None) and (int(val) == p['correcta'])


def _calificar_raiz_reportada(p, peso, marco, i):
    w = marco.f_globals.get('resp_%d' % i) if marco else None
    val = w.value if w is not None else None
    ref = p['ref']
    if val is None or ref is None:
        return 0.0
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0.0
    if abs(v - ref) <= TOL_SIMPLE * max(1.0, abs(ref)):
        return float(peso)
    return 0.0


def _calificar_caso(tema, peso, fn):
    """Califica la función del alumno contra el oráculo (Monte Carlo + eficiencia)."""
    detalle = {'aciertos': 0.0, 'iter_mediana': None,
               'iter_ref': _ref_iteraciones(tema, _EXAMEN.nc if _EXAMEN else None)}
    if fn is None:
        return 0.0, detalle
    mod = _modelo(tema)
    info = mod['info']
    lo, hi = info['intervalo']
    N = int(_RAIZ_CFG.get('N', 40))
    n_scan = int(_RAIZ_CFG.get('n_scan_oraculo', 500))
    seed = semilla(_ID, tema, _EXAMEN.nc)
    M = muestrear(tema, N=N, seed=seed)
    # La primera muestra es SIEMPRE el caso que se le mostró al alumno.
    for k, v in _params_nominales(tema, _EXAMEN.nc).items():
        if k in M:
            M[k][0] = v
    aciertos = 0
    iters = []
    for i in range(N):
        vals = {p: M[p][i] for p in mod['params']}
        rts = oraculo_escalar_todas(lambda xs: mod['f_arr'](xs, vals), lo, hi, n_scan=n_scan)
        if not rts:
            continue
        ref = rts[0]
        try:
            res = fn(lambda xv: mod['f'](xv, vals), lo, hi)
        except Exception:
            continue
        r, it = res if isinstance(res, tuple) else (res, None)
        r = _a_real(r)
        if not np.isfinite(r):
            continue
        try:
            it = int(it)
        except (TypeError, ValueError):
            it = MAX_ITER_EXAMEN
        tol_ok = max(TAU_X * max(1.0, abs(ref)), TAU_ABS)
        if abs(r - ref) <= tol_ok and it <= MAX_ITER_EXAMEN:
            aciertos += 1
            iters.append(it)
    precision = aciertos / N if N else 0.0
    if iters:
        med = float(np.median(iters))
        eficiencia = min(1.0, detalle['iter_ref'] / med) if med > 0 else 0.0
        eficiencia = max(0.2, eficiencia)
    else:
        med = None
        eficiencia = 0.0
    detalle['aciertos'] = precision
    detalle['iter_mediana'] = med
    w_pre = float(_RAIZ_CFG.get('peso_precision', 0.80))
    w_eff = float(_RAIZ_CFG.get('peso_eficiencia', 0.20))
    puntos = peso * (w_pre * precision + w_eff * eficiencia)
    return puntos, detalle


def _es_debug(nc, marco):
    es_colab = marco.f_globals.get('ES_COLAB', True) if marco else True
    return (not es_colab) and str(nc) == DEBUG_NC


def _es_colab_de(marco):
    return bool(marco.f_globals.get('ES_COLAB', False)) if marco else False


def _post_examen(payload, timeout=15):
    import requests
    return requests.post(APPS_SCRIPT_URL, json=payload, timeout=timeout).json()


def _sincronizar(alumno, marco, debug=False):
    """Estado AUTORITATIVO del servidor: {semilla, intento}.

    El servidor es quien manda: borrar archivos locales o recargar el cuaderno
    NO reinicia los intentos ni devuelve las preguntas viejas. En modo debug
    (NC del profesor en local) se omite la red para poder probar.
    Devuelve {'bloqueado': motivo} si no es hora del examen o si no hay conexión.
    """
    nc = extraer_nc(alumno)
    if debug or _es_debug(nc, marco):
        return {'semilla': _SEMILLA, 'intento': _INTENTO, 'debug': True}
    try:
        r = _post_examen({'accion': 'intento', 'token': EXAMEN_TOKEN,
                          'NC': nc, 'tarea': _ID})
        if r.get('status') == 'ok':
            d = r.get('data') or {}
            return {'semilla': int(d.get('semilla') or 1),
                    'intento': int(d.get('intento') or 0), 'debug': False}
        return {'bloqueado': str(r.get('message') or 'no se pudo verificar el intento')}
    except Exception as e:
        return {'bloqueado': 'No pude conectar con el servidor del examen (%s). '
                             'Revisa tu conexión.' % e}


def _marcar_intento(nc):
    """Suma 1 al contador de intentos de la semilla actual. Devuelve el nuevo."""
    try:
        r = _post_examen({'accion': 'intento_usado', 'token': EXAMEN_TOKEN,
                          'NC': nc, 'tarea': _ID})
        if r.get('status') == 'ok':
            d = r.get('data') or {}
            return int(d.get('intento') or 0)
    except Exception:
        pass
    return None


def _avanzar_semilla(nc):
    """Pide al servidor una NUEVA semilla (nuevo juego de teóricas)."""
    try:
        r = _post_examen({'accion': 'avanzar', 'token': EXAMEN_TOKEN, 'NC': nc,
                          'tarea': _ID, 'nPreguntas': len(_CFG['slots'])})
        if r.get('status') == 'ok':
            d = r.get('data') or {}
            return int(d.get('semilla') or 0) or None
    except Exception:
        pass
    return None


def _enviar(alumno, respuestas, pesos, puntos, persistir):
    nc = extraer_nc(alumno)
    maximo = sum(pesos)
    cuerpo = {'accion': 'examen', 'token': EXAMEN_TOKEN, 'NC': nc, 'tarea': _ID,
              'respuestas': respuestas, 'pesos': pesos, 'maxPuntos': maximo,
              'minAprobacion': MIN_APROBACION * 100}
    if not persistir:
        return cuerpo
    try:
        r = _post_examen(cuerpo, timeout=20)
        if r.get('status') != 'ok':
            print('⚠️ El servidor no aceptó el envío:', r.get('message'))
    except Exception as e:
        print('⚠️ No se pudo enviar al servidor:', e)
    return cuerpo


def _regenerar_teoria(marco):
    """Nueva ronda de preguntas TEÓRICAS (las de caso NO cambian)."""
    global _EXAMEN, _SEMILLA_EXAMEN
    nc = _EXAMEN.nc
    rng = np.random.default_rng(semilla(_ID, nc, _SEMILLA))
    _EXAMEN = Examen(nc, rng).generar()
    _SEMILLA_EXAMEN = _SEMILLA
    if marco is not None:
        for i, p in enumerate(_EXAMEN.preguntas, 1):
            if p['tipo'] == 'teorica':
                marco.f_globals.pop('resp_%d' % i, None)


def calificar(debug=False, marco=None):
    global _INTENTO, _SEMILLA, _SEMILLA_EXAMEN
    if marco is None:
        marco = inspect.currentframe().f_back
    ex = _EXAMEN
    if ex is None:
        print('Primero identifícate (generar_tarea).')
        return ResultadoCalificacion({'calificacion': 0.0, 'maximo': 0,
                                      'minimo': MIN_APROBACION * 100})
    alumno = marco.f_globals.get('alumno_id')
    # `debug` solo vale para el profesor en local: llamar calificar(debug=True)
    # con el NC de un alumno NO puede saltarse el servidor.
    debug = bool(debug) and _es_debug(extraer_nc(alumno), marco)

    # El SERVIDOR es la autoridad: relee semilla/intento antes de calificar.
    estado = _sincronizar(alumno, marco, debug=debug)
    if 'bloqueado' in estado:
        print('🚫', estado['bloqueado'])
        return ResultadoCalificacion({'calificacion': 0.0, 'maximo': ex.maximo,
                                      'minimo': MIN_APROBACION * 100,
                                      'cuerpo': {}, 'bloqueado': True})
    persistir = not estado['debug']
    if estado['semilla'] != _SEMILLA_EXAMEN:
        _SEMILLA = estado['semilla']
        _regenerar_teoria(marco)
        display(Markdown('🔄 Las preguntas teóricas cambiaron (nueva semilla %d). '
                         'Vuelve a ejecutar las celdas de las preguntas teóricas.'
                         % _SEMILLA))
    _INTENTO = estado['intento']

    n = _INTENTO + 1
    if n % 3 == 0:
        display(Markdown('**Intento %d.** ⚠️ Al terminar esta calificación, las '
                         'preguntas teóricas cambian.' % n))
    else:
        display(Markdown('**Intento %d.** Te quedan %d intento(s) antes de que '
                         'cambien las preguntas teóricas.' % (n, 3 - n % 3)))

    total = 0.0
    respuestas = []
    for i, (p, peso) in enumerate(zip(ex.preguntas, ex.pesos), 1):
        if p['tipo'] == 'teorica':
            ok = _calificar_teorica(p, marco, i)
            puntos = float(peso) if ok else 0.0
            respuestas.append(peso if ok else 0)
        elif p['tipo'] == 'raiz':
            puntos = _calificar_raiz_reportada(p, peso, marco, i)
            respuestas.append(puntos)
        elif p['tipo'] == 'caso':
            if p.get('rol') == 'desconocido':
                fn = marco.f_globals.get('raiz_otro') or marco.f_globals.get('raiz')
            else:
                fn = marco.f_globals.get('raiz')
            puntos, _ = _calificar_caso(p['tema'], peso, fn)
            respuestas.append(puntos)
        total += puntos

    maximo = ex.maximo
    minimo = MIN_APROBACION * 100
    display(Markdown('## 📝 Calificación'))
    display(Markdown('**%.1f / %d** puntos' % (total, maximo)))

    enviado = False
    cuerpo = {}
    if total >= minimo:
        cuerpo = _enviar(alumno, respuestas, ex.pesos, total, persistir)
        enviado = True
        display(Markdown('✅ **Aprobaste y tus respuestas quedaron enviadas.**'))
    else:
        display(Markdown('❌ Aún estás por debajo del **%.0f %%** (%d puntos). '
                         'Corrige y vuelve a ejecutar esta celda.'
                         % (MIN_APROBACION * 100, minimo)))

    # Persistir que se usó un intento en esta semilla (si hay servidor).
    if persistir:
        nuevo_i = _marcar_intento(alumno)
        _INTENTO = nuevo_i if nuevo_i is not None else n
    else:
        _INTENTO = n

    reiniciado = False
    if n % 3 == 0:
        if persistir:
            nueva_s = _avanzar_semilla(alumno)
            _SEMILLA = nueva_s if nueva_s else _SEMILLA + 1
        else:
            _SEMILLA += 1
        _INTENTO = 0
        _regenerar_teoria(marco)
        reiniciado = True
        display(Markdown(
            '---\n### 🔄 Se reiniciaron las preguntas teóricas\n'
            f'Has usado **{n} intentos**: las preguntas **teóricas** ahora son **otras '
            'variantes**; las de **caso** (tu caso, el otro y la raíz) siguen iguales.\n\n'
            '- Vuelve a ejecutar las celdas de las preguntas **teóricas** para ver las nuevas.\n'
            + ('- Tu calificación de este intento **ya quedó enviada**.\n' if enviado else '')
            + '- Esto pasa cada 3 intentos, para que no intentes subir la calificación '
            'sin saber que las preguntas cambian.'))

    return ResultadoCalificacion(
        {'calificacion': total, 'maximo': maximo, 'minimo': minimo,
         'cuerpo': cuerpo, 'respuestas': respuestas, 'pesos': ex.pesos,
         'intento': n, 'enviado': enviado, 'teoria_reiniciada': reiniciado,
         'semilla': _SEMILLA})


def consultar_calificacion():
    marco = inspect.currentframe().f_back
    alumno = marco.f_globals.get('alumno_id')
    if not alumno:
        print('Primero identifícate.')
        return
    import requests
    try:
        nc = extraer_nc(alumno)
        r = requests.post(APPS_SCRIPT_URL, json={
            'accion': 'consultar', 'token': WEBHOOK_TOKEN, 'NC': nc, 'tarea': _ID
        }, timeout=20).json()
        print(r.get('data') or r)
    except Exception as e:
        print('⚠️ No se pudo consultar:', e)


__all__ = ['generar_tarea', 'pregunta', 'calificar', 'consultar_calificacion',
           'extraer_nc', 'mi_caso', 'graficar_caso', 'probar_metodo']
