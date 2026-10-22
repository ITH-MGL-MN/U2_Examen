# -*- coding: utf-8 -*-
"""
profe/core/helpers.py — Utilidades base.

`_bi` (builtins) se conserva por compatibilidad con el código original de
U2_T4, que escribía `_bi.min(...)` / `_bi.max(...)` para no chocar con el
preludio estilo MATLAB del cuaderno. En el motor actual no haría falta, pero
mantenerlo hace que el catálogo y el evaluador se lean igual que antes.
"""
import builtins as _bi  # noqa: F401  (min/max/abs/any/all de Python puros)
