"""
config.py — Configuración central del proyecto
================================================
Detecta automáticamente Mac M1 (MPS), GPU NVIDIA (CUDA) o CPU.
Importa este archivo desde cualquier script del pipeline.

Uso:
    from config import DEVICE, MODELOS, TOLVA, CANDIDATOS_MATERIAL
"""

import torch
from pathlib import Path

# ──────────────────────────────────────────────────────────
# DETECCIÓN AUTOMÁTICA DE DISPOSITIVO
# ──────────────────────────────────────────────────────────

def detectar_device():
    if torch.cuda.is_available():
        dev = "cuda"
        nombre = f"GPU NVIDIA — {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available():
        dev = "mps"
        nombre = "GPU Apple Silicon (Metal/MPS)"
    else:
        dev = "cpu"
        nombre = "CPU (sin GPU)"
    print(f"✓ Dispositivo detectado: {nombre}")
    return dev

DEVICE = detectar_device()

# ──────────────────────────────────────────────────────────
# SELECCIÓN DE MODELOS
# ──────────────────────────────────────────────────────────
# Cambia "calidad" entre "rapido", "medio" o "mejor"
# para ajustar el balance velocidad/precisión.
#
# Mac M1 con 8GB RAM → recomendado: "medio"
# Mac M1 con 16GB RAM → recomendado: "mejor"
# Solo para probar que funciona → "rapido"

CALIDAD = "medio"   # ← CAMBIA ESTO

_OPCIONES_MODELOS = {
    "rapido": {
        "dino":  "IDEA-Research/grounding-dino-tiny",
        "sam2":  "facebook/sam2-hiera-tiny",
        "depth": "depth-anything/Depth-Anything-V2-Small-hf",
        "clip":  ("ViT-B-32", "openai"),
    },
    "medio": {
        "dino":  "IDEA-Research/grounding-dino-base",
        "sam2":  "facebook/sam2-hiera-small",
        "depth": "depth-anything/Depth-Anything-V2-Base-hf",
        "clip":  ("ViT-L-14", "openai"),
    },
    "mejor": {
        "dino":  "IDEA-Research/grounding-dino-base",
        "sam2":  "facebook/sam2-hiera-large",
        "depth": "depth-anything/Depth-Anything-V2-Large-hf",
        "clip":  ("ViT-L-14", "openai"),
    },
}

MODELOS = _OPCIONES_MODELOS[CALIDAD]
print(f"✓ Calidad seleccionada: {CALIDAD.upper()}")
print(f"   DINO:  {MODELOS['dino'].split('/')[-1]}")
print(f"   SAM2:  {MODELOS['sam2'].split('/')[-1]}")
print(f"   Depth: {MODELOS['depth'].split('/')[-1]}")
print(f"   CLIP:  {MODELOS['clip'][0]}")

# ──────────────────────────────────────────────────────────
# MEDIDAS REALES DE LA TOLVA
# ──────────────────────────────────────────────────────────
# Ajusta estos valores con las medidas del camión real.
# Para la prueba en oficina (caja de cartón), usa las medidas de tu caja.

TOLVA = {
    "largo_m":          5.0,    # promedio entre 4.5 y 5.5 m
    "ancho_m":          2.3,    # promedio entre 2.2 y 2.4 m
    "alto_m":           1.4,    # promedio entre 1.2 y 1.6 m
    "capacidad_m3":    16.0,    # capacidad nominal al ras
    # Para la caja de oficina (ejemplo 60x40x25 cm):
    # "largo_m":     0.60,
    # "ancho_m":     0.40,
    # "alto_m":      0.25,
    # "capacidad_m3": 0.006,
}

# ──────────────────────────────────────────────────────────
# PROMPT DE DETECCIÓN
# ──────────────────────────────────────────────────────────
# Para imagen de prueba sintética o caja de oficina:
PROMPT_TOLVA = "dump truck hopper"

# Para el camión real en obra:
# PROMPT_TOLVA = "dump truck hopper"
# PROMPT_TOLVA = "truck bed filled with material"

# Umbrales de detección (baja si no detecta nada)
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.20

# ──────────────────────────────────────────────────────────
# CANDIDATOS DE MATERIAL PARA CLIP
# ──────────────────────────────────────────────────────────
CANDIDATOS_MATERIAL = {
    "gravilla":          "crushed gravel aggregate stone texture aerial view",
    "tierra":            "compacted soil dirt earth texture top view",
    "arena":             "fine sand granular texture overhead view",
    "carbon":            "black coal crushed pieces texture top view",
    "mezcla grava+tierra": "mixed gravel and soil construction material from above",
    "escombros":         "mixed construction debris rubble from above",
    "vacia":             "empty metal container floor with rust texture",
}

# ──────────────────────────────────────────────────────────
# PATHS DE SALIDA
# ──────────────────────────────────────────────────────────
DIR_IMAGENES   = Path("imagenes_prueba")
DIR_RESULTADOS = Path("resultados")
DIR_IMAGENES.mkdir(exist_ok=True)
DIR_RESULTADOS.mkdir(exist_ok=True)
