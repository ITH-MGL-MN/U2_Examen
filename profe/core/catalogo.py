# -*- coding: utf-8 -*-
"""
profe/core/catalogo.py — Catálogo de casos de estudio (T1–T19) y familias.

Cada caso describe una ecuación (o sistema) de ingeniería con:
    'expr'/'exprs' : ecuación escalar f(x)=0  o  sistema F(x)=0
    'vars'         : variables del sistema
    'params'       : {nombre: ('n', valor, sigma_rel)}  variación normal
                     {nombre: ('u', lo, hi)}            variación uniforme
    'intervalo'    : (lo, hi) donde aislar la raíz física (escalares)
    'x0'           : punto inicial del oráculo (sistemas)
    'metodos_sugeridos' : los métodos CORRECTOS del caso (los que la
                          autocalificación exige seleccionar).
"""
import builtins as _bi


TAREA = 'U2_T4'
REPO_URL = 'https://github.com/ITH-MGL-MN/U2_T4_Casos_de_estudio.git'
REPO_NOMBRE = 'U2_T4_Casos_de_estudio'

# Tolerancias del criterio 'exito'
TAU_F = 1e-6      # |f(x)| residual
TAU_X = 1e-4      # error relativo admisible contra el oráculo
TAU_ABS = 1e-6    # error ABSOLUTO admisible = la tolerancia que se le pasa al método
MAX_ITER = 100    # iteraciones máximas permitidas
UMBRAL_EXITO = 0.80   # % de muestras que debe aprobar un método para 'contar'

CATALOGO = {
 'T1': {
   'titulo': 'Fuerza electrostática en sensor anular',
   'incognita': 'x', 'unidad': 'm',
   'ecuacion': r'F(x)=\frac{1}{4\pi\varepsilon_0}\frac{qQx}{(x^2+a^2)^{3/2}}=1\ \mathrm{N}',
   'expr': 'q*Q*x/(4*pi*e0*(x**2+a**2)**1.5) - 1',
   'params': {'e0': ('n', 8.8541878128e-12, 0.0), 'q': ('n', 1e-6, 0.02),
              'Q': ('n', 1e-6, 0.02), 'a': ('n', 0.05, 0.02)},
   'intervalo': (1e-6, 0.035),
   'metodos_sugeridos': ['Bisección', 'Newton-Raphson', 'Secante'],
   'desafio': 'La fuerza tiene un máximo en x=a/raiz(2): hay DOS ramas de solución. Se aísla la rama ascendente.',
 },
 'T2': {
   'titulo': 'Resorte no lineal en suspensión de robot móvil',
   'incognita': 'd', 'unidad': 'm',
   'ecuacion': r'\tfrac12 k_1 d^2+\tfrac25 k_2 d^{5/2}-mgd-mgh=0',
   'expr': '0.5*k1*d**2 + 0.4*k2*d**2.5 - m*g*d - m*g*h',
   'params': {'k1': ('n', 1000.0, 0.03), 'k2': ('n', 2000.0, 0.03),
              'm': ('n', 2.0, 0.03), 'g': ('n', 9.81, 0.0), 'h': ('n', 0.03, 0.03)},
   'intervalo': (0.01, 0.5),
   'metodos_sugeridos': ['Punto fijo', 'Secante', 'Brent'],
   'desafio': 'Exponente fraccionario d^{5/2}: dominio d>=0; el punto fijo puede divergir.',
 },
 'T3': {
   'titulo': 'Ángulo crítico de vuelco de robot móvil',
   'incognita': r'\theta', 'unidad': 'rad',
   'ecuacion': r'N_t=mg\cos\theta-ma\,\mathrm{sen}\,\theta-mg\frac{h}{L}\mathrm{sen}\,\theta=0',
   'expr': 'm*g*cos(th) - m*a*sin(th) - m*g*(h/L)*sin(th)',
   'params': {'m': ('n', 2.0, 0.02), 'a': ('n', 3.0, 0.02), 'h': ('n', 0.35, 0.02),
              'L': ('n', 0.5, 0.02), 'g': ('n', 9.81, 0.0)},
   'intervalo': (0.001, 1.5),
   'metodos_sugeridos': ['Bisección', 'Secante modificado', 'Brent'],
   'desafio': 'Trigonométrica: hay que filtrar la raíz física en [0, pi/2].',
 },
 'T4': {
   'titulo': 'Calibración de sensor de temperatura (polinomio)',
   'incognita': 'T', 'unidad': 'K', 'polinomico': True,
   'ecuacion': r'0.99403+1.671\!\times\!10^{-4}T+9.7215\!\times\!10^{-8}T^2-9.5838\!\times\!10^{-11}T^3+1.9520\!\times\!10^{-14}T^4=1.2',
   'expr': '0.99403 + 1.671e-4*T + 9.7215e-8*T**2 - 9.5838e-11*T**3 + 1.9520e-14*T**4 - objetivo',
   'params': {'objetivo': ('n', 1.2, 0.01)},
   'intervalo': (200.0, 3000.0),
   'metodos_sugeridos': ['Bairstow', 'Müller', 'Newton-Raphson'],
   'desafio': 'Varias raíces reales: elegir la físicamente válida (la menor positiva).',
 },
 'T5': {
   'titulo': 'Punto de operación con diodo Zener',
   'incognita': 'V_D', 'unidad': 'V',
   'ecuacion': r'I_S\!\left(e^{V_D/nV_T}-1\right)=\frac{V_{CC}-V_D}{R_L}',
   'expr': 'IS*(exp(VD/(n*VT))-1) - (VCC-VD)/RL',
   'params': {'IS': ('n', 1e-12, 0.05), 'n': ('n', 1.2, 0.05), 'VT': ('n', 0.02585, 0.05),
              'VCC': ('n', 12.0, 0.05), 'RL': ('n', 1000.0, 0.05)},
   'intervalo': (0.3, 1.0),
   'metodos_sugeridos': ['Newton-Raphson', 'Punto fijo', 'Bisección'],
   'desafio': 'La exponencial explota; el punto fijo converge lento; NR sensible a V0.',
 },
 'T6': {
   'titulo': 'Tensión en cable de catenaria',
   'incognita': 'T_A', 'unidad': 'N',
   'ecuacion': r'y=\frac{T_A}{w}\cosh\!\left(\frac{w}{T_A}x\right)+y_0-\frac{T_A}{w}',
   'expr': '(TA/w)*(cosh(w*x/TA)-1) - (y - y0)',
   'params': {'w': ('n', 20.0, 0.02), 'x': ('n', 10.0, 0.02), 'y': ('n', 15.0, 0.02),
              'y0': ('n', 0.0, 0.02)},
   'intervalo': (20.0, 5000.0),
   'metodos_sugeridos': ['Punto fijo', 'Secante modificado', 'Brent'],
   'desafio': 'cosh -> overflow si T_A es pequeño; hay que acotar bien.',
 },
 'T7': {
   'titulo': 'Tirante en canal abierto (Manning)',
   'incognita': 'H', 'unidad': 'm',
   'ecuacion': r'Q=\frac{\sqrt{S}(BH)^{5/3}}{n(B+2H)^{2/3}}',
   'expr': 'sqrt(S)*(B*H)**(5/3)/(nn*(B+2*H)**(2/3)) - Q',
   'params': {'S': ('n', 0.001, 0.03), 'Q': ('n', 5.0, 0.03), 'nn': ('n', 0.013, 0.03),
              'B': ('n', 3.0, 0.03)},
   'intervalo': (0.05, 5.0),
   'metodos_sugeridos': ['Punto fijo modificado', 'Bisección', 'Secante'],
   'desafio': 'La forma del despeje cambia la convergencia (unos divergen).',
 },
 'T8': {
   'titulo': 'Tiro parabólico con alturas desiguales',
   'incognita': r'\theta_0', 'unidad': 'rad',
   'ecuacion': r'y=(\tan\theta_0)x-\frac{g}{2v_0^2\cos^2\theta_0}x^2+y_0',
   'expr': 'g*x**2*tan(th)**2/(2*v0**2) - x*tan(th) + (y-y0) + g*x**2/(2*v0**2)',
   'params': {'x': ('n', 100.0, 0.02), 'v0': ('n', 50.0, 0.02), 'y': ('n', 0.0, 0.02),
              'y0': ('n', 10.0, 0.02), 'g': ('n', 9.81, 0.0)},
   'intervalo': (0.001, 0.5),
   'metodos_sugeridos': ['Secante modificado', 'Bisección', 'Brent'],
   'desafio': 'Dos ángulos posibles; se pide la solución de tiro tendido (raíz menor).',
 },
 'T9': {
   'titulo': 'Transitorio RLC de servomotor',
   'incognita': 'R', 'unidad': r'\Omega',
   'ecuacion': r'e^{-Rt/2L}\cos\!\Big(\sqrt{\tfrac{1}{LC}-(\tfrac{R}{2L})^2}\,t\Big)-\tfrac{q}{q_0}=0',
   'expr': 'exp(-R*t/(2*L))*cos(sqrt(1/(L*C) - (R/(2*L))**2)*t) - qq',
   'params': {'L': ('n', 0.5, 0.02), 'C': ('n', 1e-6, 0.02), 't': ('n', 4.4429e-3, 0.02),
              'qq': ('n', 0.3, 0.02)},
   'intervalo': (0.0, 800.0),
   'metodos_sugeridos': ['Bisección', 'Newton-Raphson', 'Secante'],
   'desafio': 'Oscilatoria: hay varias raíces; hay que aislar la correcta.',
 },
 'T10': {
   'titulo': 'Impedancia/resonancia en sensor piezoeléctrico',
   'incognita': r'\omega', 'unidad': 'rad/s',
   'ecuacion': r'\frac1Z=\sqrt{\frac{1}{R^2}+\left(\omega C-\frac{1}{\omega L}\right)^2}',
   'expr': '1/Z**2 - (1/R**2 + (omega*C - 1/(omega*L))**2)',
   'params': {'R': ('n', 100.0, 0.02), 'C': ('n', 1e-6, 0.02), 'L': ('n', 0.01, 0.02),
              'Z': ('n', 75.0, 0.02)},
   'intervalo': (10000.0, 30000.0),
   'metodos_sugeridos': ['Bisección', 'Falsa posición', 'Brent'],
   'desafio': 'Asíntota en w->0; la falsa posición se estanca de un lado.',
 },
 'T11': {
   'titulo': 'Volumen molar (Van der Waals)',
   'incognita': 'v', 'unidad': 'm^3/mol',
   'ecuacion': r'\left(p+\frac{a}{v^2}\right)(v-b)=RT',
   'expr': '(p + a/v**2)*(v - b) - R*T',
   'estimacion': 'R*T/p',
   'params': {'p': ('n', 1e6, 0.02), 'T': ('n', 300.0, 0.02), 'a': ('n', 0.3640, 0.02),
              'b': ('n', 4.267e-5, 0.02), 'R': ('n', 8.314, 0.0)},
   'intervalo': (2e-4, 1e-2),
   'metodos_sugeridos': ['Newton-Raphson', 'Secante', 'Brent'],
   'desafio': 'Comparar con el gas ideal; sensibilidad a la presión.',
 },
 'T12': {
   'titulo': 'Razón de recirculación en reactor autocatalítico',
   'incognita': 'R', 'unidad': '-',
   'ecuacion': r'\ln\frac{1+R(1-X_{Af})}{R(1-X_{Af})}=\frac{R+1}{R\,[1+R(1-X_{Af})]}',
   'expr': 'log((1+Rr*(1-Xaf))/(Rr*(1-Xaf))) - (Rr+1)/(Rr*(1+Rr*(1-Xaf)))',
   'params': {'Xaf': ('n', 0.9, 0.02)},
   'intervalo': (0.01, 1.0),
   'metodos_sugeridos': ['Bisección', 'Punto fijo', 'Secante'],
   'desafio': 'Discontinuidad del logaritmo cerca de R->0.',
 },
 'T13': {
   'titulo': 'Colebrook + caída de presión',
   'incognita': 'f', 'unidad': '-',
   'ecuacion': r'\frac{1}{\sqrt f}=-2\log_{10}\!\left(\frac{\epsilon}{3.7D}+\frac{2.51}{Re\sqrt f}\right),\ Re=\frac{\rho VD}{\mu}',
   'expr': '1/sqrt(f) + 2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f)))',
   'params': {'eps': ('n', 4.5e-5, 0.02), 'D': ('n', 0.05, 0.02), 'Re': ('n', 66666.667, 0.02)},
   'intervalo': (0.005, 0.2),
   'metodos_sugeridos': ['Punto fijo', 'Newton-Raphson', 'Brent'],
   'desafio': 'NR diverge si x0 > 0.066; usar Swamee-Jain como x0.',
 },
 'T16': {
   'titulo': 'Velocidad terminal con C_D no lineal',
   'incognita': 'v', 'unidad': 'm/s',
   'ecuacion': r'v=\sqrt{\frac{4g(\rho_s-\rho)d}{3C_D\rho}},\ C_D=\frac{24}{Re}+\frac{3}{\sqrt{Re}}+0.34',
   'expr': '3*rho*v**2*(24*mu/(rho*d*v) + 3*sqrt(mu/(rho*d*v)) + 0.34) - 4*g*(rhos-rho)*d',
   'params': {'d': ('n', 0.002, 0.03), 'rhos': ('n', 2500.0, 0.03), 'rho': ('n', 1.2, 0.03),
              'mu': ('n', 1.8e-5, 0.03), 'g': ('n', 9.81, 0.0)},
   'intervalo': (1.0, 60.0),
   'metodos_sugeridos': ['Secante', 'Secante modificado', 'Brent'],
   'desafio': 'Verificar el régimen resultante (Re > 0.1).',
 },
 'T17': {
   'titulo': 'Péndulo amortiguado forzado (amplitud/resonancia)',
   'incognita': r'\omega', 'unidad': 'rad/s',
   'ecuacion': r'(\omega_n^2-\omega^2)^2+(2\zeta\omega_n\omega)^2=(F_0/mA)^2',
   'expr': '(wn**2 - omega**2)**2 + (2*z*wn*omega)**2 - (F0/(m*A))**2',
   'params': {'wn': ('n', 10.0, 0.03), 'z': ('n', 0.05, 0.03), 'F0': ('n', 1.0, 0.03),
              'm': ('n', 1.0, 0.03), 'A': ('n', 0.01, 0.03)},
   'intervalo': (1.0, 30.0),
   'metodos_sugeridos': ['Newton-Raphson', 'Secante modificado', 'Brent'],
   'desafio': 'Seleccionar la raíz física (w > 0, por encima de la resonancia).',
 },
 'T18': {
   'titulo': 'Empuje vectorizado (gimbal) — balance de momentos',
   'incognita': r'\theta', 'unidad': 'rad',
   'ecuacion': r'(2T_b a+T_o b)\cos\theta-W_S(c\,\mathrm{sen}\,\theta+d\cos\theta)=0',
   'expr': '(2*Tb*a + To*b)*cos(th) - Ws*(c*sin(th) + d*cos(th))',
   'params': {'Tb': ('n', 1.5e5, 0.03), 'To': ('n', 4.0e4, 0.03), 'a': ('n', 50.0, 0.0),
              'b': ('n', 80.0, 0.0), 'c': ('n', 120.0, 0.0), 'd': ('n', 20.0, 0.0),
              'Ws': ('u', 195000.0, 230000.0)},
   'intervalo': (-0.5, 1.5),
   'metodos_sugeridos': ['Newton-Raphson', 'Brent', 'Secante'],
   'desafio': 'Varias raíces; graficar M(theta) en [-pi, pi]; sensibilidad al peso.',
 },
 'T14': {
   'titulo': 'Cinemática inversa 2-DOF (Jacobiano)',
   'incognita': r'(\theta_1,\theta_2)', 'unidad': 'rad', 'sistema': True,
   'vars': ['th1', 'th2'],
   'ecuacion': r'L_1\cos\theta_1+L_2\cos(\theta_1+\theta_2)=x;\ \ L_1\mathrm{sen}\,\theta_1+L_2\mathrm{sen}(\theta_1+\theta_2)=y',
   'exprs': ['L1*cos(th1) + L2*cos(th1+th2) - x',
             'L1*sin(th1) + L2*sin(th1+th2) - y'],
   'params': {'L1': ('n', 0.3, 0.01), 'L2': ('n', 0.25, 0.01),
              'x': ('n', 0.35, 0.01), 'y': ('n', 0.2, 0.01)},
   'x0': [0.5, 0.5],
   'metodos_sugeridos': ['Newton-Raphson multivariable', 'Punto fijo modificado'],
   'desafio': 'Jacobiano singular en la frontera del espacio de trabajo.',
 },
 'T15': {
   'titulo': 'Multilateración con 3 balizas (Jacobiano)',
   'incognita': '(x,y)', 'unidad': 'm', 'sistema': True,
   'vars': ['x', 'y'],
   'ecuacion': r'(x-x_i)^2+(y-y_i)^2=d_i^2,\ i=1,2,3',
   'exprs': ['x**2 + y**2 - d1**2',
             '(x-1)**2 + y**2 - d2**2'],
   'params': {'d1': ('n', 0.54083, 0.01), 'd2': ('n', 0.83217, 0.01)},
   'x0': [0.2, 0.2],
   'metodos_sugeridos': ['Newton-Raphson multivariable'],
   'desafio': 'Sistema con ruido de medición (sigma=1%); estimación inicial crítica.',
 },
 'T19': {
   'titulo': 'Red de tuberías no lineal (2-3 nodos)',
   'incognita': '(Q_1,Q_2)', 'unidad': 'm^3/s', 'sistema': True,
   'vars': ['Q1', 'Q2'],
   'ecuacion': r'\sum Q_i=0;\ \ \Delta p=K Q^n\ \text{por tramo}',
   'exprs': ['K1*Q1**n + K2*Q2**n + K3*abs(Q1-Q2)**n - dH',
             'K2*Q2**n - K3*abs(Q1-Q2)**n'],
   'params': {'K1': ('n', 1000.0, 0.03), 'K2': ('n', 5000.0, 0.03), 'K3': ('n', 1000.0, 0.03),
              'n': ('n', 2.0, 0.03), 'dH': ('n', 50.0, 0.03)},
   'x0': [0.15, 0.05],
   'metodos_sugeridos': ['Newton-Raphson multivariable', 'Punto fijo modificado'],
   'desafio': 'Acoplamiento entre nodos y buen punto inicial.',
 },
}


# ===========================================================================
# Familias de métodos
# ===========================================================================
FAMILIA_DE = {
    'Bisección': 'cerrado', 'Falsa posición': 'cerrado', 'Punto fijo modificado': 'cerrado',
    'Newton-Raphson': 'abierto', 'Newton-Raphson multivariable': 'abierto',
    'Punto fijo': 'abierto', 'NR modificado': 'abierto', 'Secante': 'abierto',
    'Secante modificado': 'abierto', 'Brent': 'hibrido',
    'Müller': 'polinomios', 'Bairstow': 'polinomios',
}
_ALIAS = {
    'biseccion': 'Bisección', 'falsa posicion': 'Falsa posición', 'falsa_posicion': 'Falsa posición',
    'punto fijo modificado': 'Punto fijo modificado', 'pfm': 'Punto fijo modificado',
    'punto fijo': 'Punto fijo', 'punto_fijo': 'Punto fijo',
    'newton raphson': 'Newton-Raphson', 'newton_raphson': 'Newton-Raphson',
    'newton': 'Newton-Raphson', 'nr': 'Newton-Raphson',
    'newton raphson multivariable': 'Newton-Raphson multivariable', 'newton multivariable': 'Newton-Raphson multivariable',
    'newton_sistema': 'Newton-Raphson multivariable', 'newton sistema': 'Newton-Raphson multivariable',
    'newton_sistema_2x2': 'Newton-Raphson multivariable', 'newton_multivariable': 'Newton-Raphson multivariable',
    'punto_fijo_sistema': 'Punto fijo modificado',
    'nr modificado': 'NR modificado', 'nr_modificado': 'NR modificado',
    'secante': 'Secante', 'secante modificado': 'Secante modificado',
    'secante_modificado': 'Secante modificado', 'secante_modificada': 'Secante modificado',
    'brent': 'Brent', 'muller': 'Müller', 'müller': 'Müller', 'bairstow': 'Bairstow',
}


def _norm(nombre):
    """Nombre canónico del método (o el original si no se reconoce)."""
    s = str(nombre).strip().lower().replace('ó', 'o').replace('ú', 'u').replace('á', 'a')
    return _ALIAS.get(s, str(nombre).strip())


def familia(nombre):
    """Familia canónica de un método (cerrado/abierto/hibrido/polinomios/desconocida)."""
    return FAMILIA_DE.get(_norm(nombre), 'desconocida')


def familia_de(nombre):
    """Alias de `familia` para mayor claridad."""
    return familia(nombre)


def es_sistema(tema):
    """¿El caso es un sistema de ecuaciones? (usa Jacobiano, no intervalo)."""
    return bool(CATALOGO[tema].get('sistema'))


def metodos_sugeridos_para(tema):
    """Los métodos CORRECTOS (sugeridos) del caso, ya normalizados."""
    return [_norm(m) for m in CATALOGO[tema]['metodos_sugeridos']]


def detectar_metodos(espacio, nombres_declarados=()):
    """Escaneo con inspect: nombres normalizados presentes en el código del alumno.

    Solo sirve para AVISAR (no califica). Devuelve (encontrados, avisos).
    """
    import inspect
    declarados = {_norm(n) for n in nombres_declarados}
    encontrados = set()
    for nombre, obj in list(espacio.items()):
        if nombre.startswith('_'):
            continue
        if not (inspect.isfunction(obj) or callable(obj)):
            continue
        cand = _norm(nombre)
        if cand in FAMILIA_DE:
            encontrados.add(cand)
    avisos = []
    for n in sorted(encontrados - declarados):
        avisos.append('Detecté "%s" en tu código pero no está en METODOS; agrégalo para que cuente.' % n)
    for n in sorted(declarados - encontrados):
        avisos.append('"%s" está declarado en METODOS pero no encontré la función.' % n)
    return sorted(encontrados), avisos
