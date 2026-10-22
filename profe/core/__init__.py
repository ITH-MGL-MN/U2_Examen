# -*- coding: utf-8 -*-
# profe/core/__init__.py — Reexporta la API del motor para `from profe.core import ...`.
from .catalogo import (
    CATALOGO,
    FAMILIA_DE,
    MAX_ITER,
    TAREA,
    REPO_URL,
    REPO_NOMBRE,
    TAU_X,
    TAU_ABS,
    UMBRAL_EXITO,
    _norm,
    familia,
    familia_de,
    es_sistema,
    metodos_sugeridos_para,
    detectar_metodos,
)
from .seed import extraer_nc, semilla
from .modelos import modelo, _modelo
from .oraculo import oraculo_escalar, oraculo_escalar_todas, oraculo_sistema
from .metodos import mapa_metodos
from .montecarlo import muestrear, evaluar_tema, MOTIVOS_TEXTO

__all__ = [
    'CATALOGO', 'FAMILIA_DE', 'MAX_ITER', 'TAREA', 'TAU_X', 'TAU_ABS', 'TAU_F',
    'UMBRAL_EXITO', 'MOTIVOS_TEXTO', '_norm', 'familia', 'familia_de',
    'es_sistema', 'metodos_sugeridos_para', 'detectar_metodos',
    'extraer_nc', 'semilla', 'modelo', 'oraculo_escalar',
    'oraculo_escalar_todas', 'oraculo_sistema', 'mapa_metodos',
    'muestrear', 'evaluar_tema',
]
