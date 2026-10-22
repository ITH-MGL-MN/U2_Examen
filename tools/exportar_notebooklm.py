#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exportar_notebooklm.py - pasa TODA la tarea a archivos .txt legibles.

Sirve para cargar el proyecto como fuentes en Google NotebookLM (o en cualquier
otro asistente que coma texto plano).  Un archivo de entrada = un archivo .txt
de salida, con un encabezado que dice de donde salio cada cosa.

Que hace con cada tipo de archivo
---------------------------------
  .ipynb  ->  texto legible: las celdas markdown tal cual y las celdas de
              codigo dentro de un bloque ```python.  Las SALIDAS (outputs) se
              descartan: son ruido y no aportan al resumen.
  otros   ->  el contenido tal cual (.py, .md, .yaml, .csv, .txt, .gs, ...).

Que NO exporta
--------------
  * grader_ofuscado.txt : es el blob base64+zlib, ilegible a proposito.
  * grader_local.py     : es el bundle GENERADO por tools/build.py y su
                          contenido ya esta en profe/**; duplicarlo ensucia el
                          buscador.  Usa --incluir-bundle para agregarlo.
  * config/config.yaml  : lleva el token del webhook y la URL de Apps Script;
                          no aporta nada al material de estudio.
  * carpetas de trabajo : __pycache__, .git, .ipynb_checkpoints y la propia
                          carpeta de salida.

Uso
---
    python tools/exportar_notebooklm.py              # -> <tarea>/notebooklm/
    python tools/exportar_notebooklm.py --unir       # + 00_TODO_U2_T4.txt
    python tools/exportar_notebooklm.py --incluir-bundle
    python tools/exportar_notebooklm.py -o C:\\tmp\\mn

Con --unir se genera, ademas de los archivos sueltos, UN SOLO .txt con todo
el material seguido: es el que se sube como una unica fuente a NotebookLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# tools/ -> raiz de la tarea.  El script funciona igual si lo copias a la
# carpeta tools/ de otra tarea del curso.
RAIZ = Path(__file__).resolve().parent.parent
NOMBRE_PROYECTO = RAIZ.name
# Nombre corto de la tarea para el archivo unido (U2_Examen ->
# 00_TODO_U2_Examen.txt), que asi se llama igual que el cuaderno.
NOMBRE_TAREA = "_".join(NOMBRE_PROYECTO.split("_")[:2])
NOMBRE_UNIDO = "00_TODO_%s.txt" % NOMBRE_TAREA

# --- Que se lee -------------------------------------------------------
EXTENSIONES = {
    ".py", ".md", ".txt", ".ipynb", ".yaml", ".yml", ".json",
    ".csv", ".gs", ".js", ".tex", ".html", ".css",
}
SIN_EXTENSION = {".gitignore", ".gitattributes", "LICENSE"}

# --- Que se ignora ----------------------------------------------------
CARPETAS_IGNORADAS = {
    "__pycache__", ".git", ".ipynb_checkpoints", ".venv", "venv",
    "node_modules", ".vscode", ".idea", ".mypy_cache", ".pytest_cache",
    # la salida por omision: si se usa -o, la carpeta activa se ignora aparte,
    # pero esto evita re-leer un export anterior y duplicar las fuentes
    "notebooklm",
}
ARCHIVOS_IGNORADOS = {
    "grader_ofuscado.txt",   # el blob ofuscado: ilegible a proposito
    "grader_local.py",       # bundle generado; ya esta en profe/**
    ".gitignore",            # ruido de configuracion
}
# Rutas exactas (no solo el nombre) que tampoco se exportan.
RUTAS_IGNORADAS = {
    "config/config.yaml",    # token del webhook + URL de Apps Script
}
BUNDLE = "grader_local.py"

SEPARADOR = "=" * 70


# ---------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------
def leer_texto(ruta: Path):
    """Devuelve (texto, codec) o (None, motivo) si el archivo es binario."""
    datos = ruta.read_bytes()
    if b"\0" in datos[:8192]:
        return None, "binario"
    try:
        return datos.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return datos.decode("latin-1"), "latin-1"


def ipynb_a_texto(ruta: Path) -> str:
    """Convierte un cuaderno a texto: markdown + codigo, SIN salidas."""
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    kernel = (nb.get("metadata", {}).get("kernelspec") or {}).get("name") or "python"
    partes = []
    for i, celda in enumerate(nb.get("cells", []), start=1):
        tipo = celda.get("cell_type", "?")
        fuente = celda.get("source", "")
        if isinstance(fuente, list):
            fuente = "".join(fuente)
        fuente = fuente.replace("\r\n", "\n").rstrip("\n")
        if not fuente.strip():
            continue
        if tipo == "markdown":
            partes.append("----- Celda %d  (markdown) -----\n\n%s" % (i, fuente))
        elif tipo == "code":
            partes.append("----- Celda %d  (codigo) -----\n\n```%s\n%s\n```"
                          % (i, kernel, fuente))
        elif tipo == "raw":
            partes.append("----- Celda %d  (raw) -----\n\n%s" % (i, fuente))
    return "\n\n".join(partes)


# ---------------------------------------------------------------------
# Recoleccion
# ---------------------------------------------------------------------
def recolectar(salida: Path, incluir_bundle: bool, max_bytes: int):
    """Devuelve ([(ruta, relativa)], [(relativa, motivo)]) de lo exportable."""
    ignorados = set(ARCHIVOS_IGNORADOS)
    if incluir_bundle:
        ignorados.discard(BUNDLE)

    encontrados, saltados = [], []
    for ruta in sorted(RAIZ.rglob("*"), key=lambda p: p.as_posix().lower()):
        if not ruta.is_file():
            continue
        rel = ruta.relative_to(RAIZ)
        if any(p in CARPETAS_IGNORADAS for p in rel.parts):
            continue
        # la carpeta de salida puede estar dentro de la tarea: no la re-leemos
        if salida == ruta or salida in ruta.parents:
            continue
        if rel.as_posix() in RUTAS_IGNORADAS:
            saltados.append((rel, "excluido (lleva credenciales)"))
            continue
        if ruta.name in ignorados:
            saltados.append((rel, "excluido por nombre"))
            continue
        if ruta.name not in SIN_EXTENSION and ruta.suffix.lower() not in EXTENSIONES:
            saltados.append((rel, "extension no reconocida"))
            continue
        if ruta.stat().st_size > max_bytes:
            saltados.append((rel, "demasiado grande (>%.0f KB)" % (max_bytes / 1024.0)))
            continue
        encontrados.append((ruta, rel))
    return encontrados, saltados


# ---------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------
def cabecera(rel: Path, texto: str, nota: str = "") -> str:
    kb = len(texto.encode("utf-8")) / 1024.0
    lineas = [
        SEPARADOR,
        "ARCHIVO : %s" % rel.as_posix(),
        "PROYECTO: %s" % NOMBRE_PROYECTO,
        "TAMANO  : %.1f KB   LINEAS: %d" % (kb, texto.count("\n") + 1),
    ]
    if nota:
        lineas.append("NOTA    : %s" % nota)
    lineas += [SEPARADOR, "", ""]
    return "\n".join(lineas)


def nombre_txt(rel: Path) -> str:
    """profe/core/modelos.py -> profe__core__modelos.py.txt (plano y unico)."""
    return "__".join(rel.parts) + ".txt"


def limpiar_salida(salida: Path) -> int:
    """Borra solo los .txt generados antes; deja cualquier otra cosa intacta."""
    if not salida.is_dir():
        return 0
    n = 0
    for viejo in salida.glob("*.txt"):
        viejo.unlink()
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Exporta la tarea a .txt para NotebookLM.")
    ap.add_argument("-o", "--salida", default=None,
                    help="carpeta destino (por omision: <tarea>/notebooklm)")
    ap.add_argument("--incluir-bundle", action="store_true",
                    help="incluye tambien grader_local.py (bundle generado)")
    ap.add_argument("--unir", action="store_true",
                    help="ademas de los archivos sueltos, uno solo con todo")
    ap.add_argument("--max-kb", type=int, default=1500,
                    help="ignora archivos mayores a esto, en KB (por omision 1500)")
    args = ap.parse_args(argv)

    salida = Path(args.salida).expanduser().resolve() if args.salida else RAIZ / "notebooklm"
    encontrados, saltados = recolectar(salida, args.incluir_bundle, args.max_kb * 1024)

    if not encontrados:
        print("No encontre archivos que exportar en %s" % RAIZ)
        return 1

    salida.mkdir(parents=True, exist_ok=True)
    borrados = limpiar_salida(salida)

    print("Origen : %s" % RAIZ)
    print("Destino: %s" % salida)
    if borrados:
        print("         (%d .txt previos borrados)" % borrados)
    print()
    print("%-11s %-8s %s" % ("TAMANO", "CODIF", "ARCHIVO"))
    print("-" * 70)

    registros = []          # (rel, nombre_txt, bytes, codec)
    for ruta, rel in encontrados:
        if ruta.suffix.lower() == ".ipynb":
            try:
                texto = ipynb_a_texto(ruta)
            except Exception as exc:                       # noqa: BLE001
                print("  !! %s: no pude leer el cuaderno (%r)" % (rel.as_posix(), exc))
                continue
            codec, nota = "json", "cuaderno convertido a texto (sin salidas)"
        else:
            texto, codec = leer_texto(ruta)
            if texto is None:
                print("  -- %s: %s, se salta" % (rel.as_posix(), codec))
                continue
            nota = "" if codec == "utf-8" else "codificado en %s" % codec

        texto = texto.replace("\r\n", "\n")
        cuerpo = cabecera(rel, texto, nota) + texto + "\n"
        destino = salida / nombre_txt(rel)
        destino.write_text(cuerpo, encoding="utf-8")

        kb = len(cuerpo.encode("utf-8")) / 1024.0
        registros.append((rel, destino.name, kb, codec))
        print("%7.1f KB  %-8s %s" % (kb, codec, destino.name))

    # --- Indice -------------------------------------------------------
    total = sum(r[2] for r in registros)
    lineas = [
        SEPARADOR,
        "INDICE DE FUENTES - %s" % NOMBRE_PROYECTO,
        SEPARADOR,
        "",
        "%d archivos exportados, %.0f KB en total." % (len(registros), total),
        "",
        "Cada .txt empieza con el nombre del archivo original, asi que al citar",
        "en NotebookLM siempre se ve de donde salio el texto.",
        "",
        "Version en un solo archivo:  python tools/exportar_notebooklm.py --unir",
        "                              -> %s" % NOMBRE_UNIDO,
        "",
        "Generado con:  python tools/exportar_notebooklm.py",
        "",
        "-" * 70,
        "%-4s %-52s %8s" % ("#", "ARCHIVO ORIGINAL  ->  .txt", "TAMANO"),
        "-" * 70,
    ]
    for i, (rel, nombre, kb, _codec) in enumerate(registros, start=1):
        etiqueta = "%s  ->  %s" % (rel.as_posix(), nombre)
        lineas.append("%-4d %-52s %6.1f KB" % (i, etiqueta[:52], kb))
    lineas += ["-" * 70, ""]

    if saltados:
        lineas += ["NO EXPORTADOS", ""]
        for rel, motivo in saltados:
            lineas.append("  %-42s %s" % (rel.as_posix(), motivo))
        lineas += ["", "(grader_ofuscado.txt es el blob ofuscado; grader_local.py es el",
                   " bundle generado por tools/build.py y ya esta en profe/**;",
                   " config/config.yaml lleva el token del webhook.)", ""]

    (salida / "00_INDICE.txt").write_text("\n".join(lineas), encoding="utf-8")

    if args.unir:
        partes = [(salida / nombre).read_text(encoding="utf-8")
                  for _rel, nombre, _kb, _codec in registros]
        (salida / NOMBRE_UNIDO).write_text(
            "\n\n".join(partes), encoding="utf-8")

    print("-" * 70)
    print("%d archivos, %.0f KB.  Indice: %s"
          % (len(registros), total, (salida / "00_INDICE.txt").name))
    if args.unir:
        kb_unido = (salida / NOMBRE_UNIDO).stat().st_size / 1024.0
        print("Version unica: %s (%.0f KB)" % (NOMBRE_UNIDO, kb_unido))
    print()
    print("En NotebookLM: 'Agregar fuente > Subir archivos' y eliges la carpeta")
    print("%s completa (límite de 50 fuentes por cuaderno), o el archivo unido."
          % salida.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
