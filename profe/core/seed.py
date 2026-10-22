# -*- coding: utf-8 -*-
"""
profe/core/seed.py — Semillas deterministas y limpieza del Número de Control.

U2_T4 NO usa semilla por alumno (el equipo elige su caso), pero conserva
`semilla(tema, nc_jefe)` para que la simulación de Monte Carlo sea
REPRODUCIBLE: el mismo equipo, con el mismo tema, obtiene exactamente las
mismas muestras y las mismas gráficas.
"""
import hashlib
import re


def extraer_nc(identificador):
    """
    Extrae únicamente la secuencia de dígitos del Número de Control (NC) a
    partir del correo institucional o identificador.

    Formatos aceptados:
      - 'm16330887@hermosillo.tecnm.mx' -> '16330887'
      - '16330887@hermosillo.tecnm.mx'  -> '16330887'
      - 'm{16330887}@tecnm.mx'          -> '16330887'
    """
    if identificador is None:
        return None
    s = str(identificador).strip()
    match_llaves = re.search(r'\{([^}]+)\}', s)
    if match_llaves:
        return match_llaves.group(1).strip()
    local = s.split('@')[0]
    match_digitos = re.search(r'\d{4,}', local)
    if match_digitos:
        return match_digitos.group(0)
    return None


def semilla(*partes):
    """Semilla determinista y reproducible (nunca usar hash())."""
    texto = '|'.join(str(p) for p in partes).encode('utf-8')
    return int(hashlib.sha256(texto).hexdigest(), 16) % 2 ** 32
