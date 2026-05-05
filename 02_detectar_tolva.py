"""
02 - DETECCION Y SEGMENTACION DE LA TOLVA
==========================================
Estrategia:
  PASO A: SAM 2 con solo bbox → candidato mayor score
          → segmenta el BORDE de la tolva (hueco negro = material)
  PASO B: INVERTIR esa máscara → hueco negro se convierte en material
  PASO C: Recorte geométrico asimétrico → elimina cabina y bordes
          (más recorte arriba que abajo porque el borde superior
           tiene más hueco que el inferior)

Compatible con Mac M1/M2/M3 (MPS), GPU NVIDIA (CUDA) y CPU.

Uso:
    python3 02_detectar_tolva.py imagenes_prueba/tolva_llena.png

Ajuste fino al inicio del script:
    CORTE_CABINA_PCT    → sube si entra la cabina
    MARGEN_SUPERIOR_PCT → sube si entra el borde superior metálico
    MARGEN_INFERIOR_PCT → sube si entra el borde inferior metálico
    MARGEN_FRONTAL_PCT  → sube si entra el borde frontal derecho
"""

import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# ── Device ────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SAM_DEVICE = "cpu" if DEVICE == "mps" else DEVICE
print(f"✓ Device: {DEVICE.upper()}  |  SAM2: {SAM_DEVICE.upper()}")

# ── Config ────────────────────────────────────────────────
try:
    from config import MODELOS, PROMPT_TOLVA, BOX_THRESHOLD, \
                       TEXT_THRESHOLD, DIR_RESULTADOS
    DINO_MODEL_ID = MODELOS["dino"]
    SAM2_MODEL_ID = MODELOS["sam2"]
    PROMPT_ACTIVO = PROMPT_TOLVA
except ImportError:
    DINO_MODEL_ID  = "IDEA-Research/grounding-dino-base"
    SAM2_MODEL_ID  = "facebook/sam2-hiera-small"
    PROMPT_ACTIVO  = "dump truck hopper"
    BOX_THRESHOLD  = 0.30
    TEXT_THRESHOLD = 0.20
    DIR_RESULTADOS = Path("resultados")
    DIR_RESULTADOS.mkdir(exist_ok=True)

PROMPTS_ALTERNATIVOS = [
    "dump truck hopper",
    "truck bed filled with material",
    "dump truck bed",
    "dump truck",
    "yellow truck",
]

# ──────────────────────────────────────────────────────────
# PARÁMETROS DE AJUSTE FINO
# ──────────────────────────────────────────────────────────

# % del bbox que ocupa la cabina (se descarta por la izquierda)
CORTE_CABINA_PCT    = 0.32

# % del alto del bbox para el borde SUPERIOR (más agresivo porque
# el borde superior tiene un hueco que conecta con el exterior)
MARGEN_SUPERIOR_PCT = 0.12

# % del alto del bbox para el borde INFERIOR
MARGEN_INFERIOR_PCT = 0.06

# % del ancho para el borde frontal derecho de la tolva
MARGEN_FRONTAL_PCT  = 0.02

# Tamaño del kernel de closing para rellenar huecos de piedras
# Valor pequeño = solo rellena piedras individuales
# Valor grande = rellena huecos grandes (pero puede conectar bordes)
KERNEL_CLOSING = 20


# ── Grounding DINO ────────────────────────────────────────

def cargar_grounding_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print(f"\nCargando Grounding DINO — {DINO_MODEL_ID.split('/')[-1]}...")
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection\
              .from_pretrained(DINO_MODEL_ID).to(DEVICE).eval()
    print(f"✓ Grounding DINO listo  [{DEVICE.upper()}]")
    return processor, model


def detectar_bbox(img_pil, processor, model, prompt,
                  box_threshold=BOX_THRESHOLD,
                  text_threshold=TEXT_THRESHOLD):
    if not prompt.endswith("."):
        prompt = prompt + "."
    inputs = processor(images=img_pil, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    try:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[img_pil.size[::-1]]
        )[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=box_threshold,
            target_sizes=[img_pil.size[::-1]]
        )[0]
    if len(res["boxes"]) == 0:
        return None, 0.0
    idx = res["scores"].argmax().item()
    return res["boxes"][idx].cpu().numpy().astype(int), res["scores"][idx].item()


# ── SAM 2 ─────────────────────────────────────────────────

def cargar_sam2():
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"\nCargando SAM 2 — {SAM2_MODEL_ID.split('/')[-1]}...")
    predictor = SAM2ImagePredictor.from_pretrained(
        SAM2_MODEL_ID, device=SAM_DEVICE)
    print(f"✓ SAM 2 listo  [{SAM_DEVICE.upper()}]")
    return predictor


def segmentar_borde_tolva(img_np_rgb, bbox, predictor):
    """
    PASO A: SAM 2 solo bbox, sin puntos.
    Elige el candidato de MAYOR SCORE (~0.91) que segmenta
    el borde metálico de la tolva con el hueco negro = material.
    """
    with torch.no_grad():
        predictor.set_image(img_np_rgb)
        mascaras, scores, _ = predictor.predict(
            box=np.array(bbox),
            multimask_output=True
        )

    print(f"  Candidatos:")
    for i, (s, m) in enumerate(zip(scores, mascaras)):
        print(f"    {i+1}: score={s:.3f}  área={m.mean()*100:.1f}%")

    idx_mejor = int(np.argmax(scores))
    mascara   = mascaras[idx_mejor].astype(bool)
    score     = float(scores[idx_mejor])
    print(f"  → Usando candidato {idx_mejor+1} "
          f"(score={score:.3f}  área={mascara.mean()*100:.1f}%)")

    return mascara, score


def extraer_material(mascara_borde, bbox, img_shape):
    """
    PASO B+C:
    1. Invertir la máscara dentro del bbox → hueco negro = material
    2. Recorte geométrico ASIMÉTRICO:
       - Más recorte arriba (MARGEN_SUPERIOR_PCT) para eliminar el
         hueco del borde superior que conecta con el exterior
       - Menos recorte abajo (MARGEN_INFERIOR_PCT)
    3. Closing pequeño → rellena huecos de piedras individuales
    4. Componente más grande → elimina ruido
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    # Límites de la región interior (asimétrico arriba/abajo)
    xi = int(x1 + bw * CORTE_CABINA_PCT)
    xf = int(x2 - bw * MARGEN_FRONTAL_PCT)
    yi = int(y1 + bh * MARGEN_SUPERIOR_PCT)   # más margen arriba
    yf = int(y2 - bh * MARGEN_INFERIOR_PCT)   # menos margen abajo

    print(f"\n  Región interior:")
    print(f"    X: {xi} → {xf}  ({xf-xi} px)")
    print(f"    Y: {yi} → {yf}  ({yf-yi} px)")
    print(f"    Margen superior: {MARGEN_SUPERIOR_PCT:.0%}  "
          f"inferior: {MARGEN_INFERIOR_PCT:.0%}")

    # PASO B: invertir dentro del bbox → False (negro) = material
    m_invertida = np.zeros(img_shape[:2], dtype=np.uint8)
    region_bbox = mascara_borde[y1:y2, x1:x2]
    m_invertida[y1:y2, x1:x2] = (~region_bbox).astype(np.uint8)

    # PASO C: recorte geométrico asimétrico
    m_recortada = np.zeros_like(m_invertida)
    m_recortada[yi:yf, xi:xf] = m_invertida[yi:yf, xi:xf]

    # Closing pequeño: rellena huecos de piedras pero no conecta bordes
    kernel_close = np.ones((KERNEL_CLOSING, KERNEL_CLOSING), np.uint8)
    m_filled = cv2.morphologyEx(
        m_recortada * 255, cv2.MORPH_CLOSE, kernel_close)

    # Componente conectado más grande
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
        m_filled, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx_mayor = np.argmax(areas) + 1
        m_final = (labels_cc == idx_mayor).astype(np.uint8) * 255
    else:
        m_final = m_filled

    mascara_final = m_final > 127

    print(f"  Material final: {mascara_final.sum():,} px  "
          f"({mascara_final.mean()*100:.1f}% imagen)")

    return mascara_final, (xi, yi, xf, yf)


# ── Visualización ─────────────────────────────────────────

def visualizar(img_np, bbox, mascara_borde, mascara_material,
               score_bbox, score_sam, region, ruta_salida):

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Paso 1 — Detección y segmentación (solo material interior)",
                  color="#E8EAF6", fontsize=13, fontweight="bold")
    T="#E8EAF6"; L="#90A4AE"; PAN="#1a1d27"
    for ax in axes:
        ax.set_facecolor(PAN)
        [sp.set_color("#2d3249") for sp in ax.spines.values()]

    x1, y1, x2, y2 = bbox
    xi, yi, xf, yf = region

    # Panel 1: imagen + bbox + región material
    axes[0].imshow(img_np)
    axes[0].set_title(f"Detección DINO  (score: {score_bbox:.2f})",
                       color=T, fontsize=10)
    axes[0].add_patch(patches.Rectangle(
        (x1,y1), x2-x1, y2-y1,
        lw=2, edgecolor="#40C4FF", facecolor="none", ls="--",
        label="bbox camión"))
    axes[0].add_patch(patches.Rectangle(
        (xi,yi), xf-xi, yf-yi,
        lw=2.5, edgecolor="#69F0AE", facecolor="none", ls="-",
        label="región material"))
    axes[0].legend(fontsize=8, facecolor=PAN, labelcolor=L, loc="lower right")
    axes[0].axis("off")

    # Panel 2: máscara SAM 2 borde (naranja) con hueco (negro = material)
    mv_borde = np.zeros((*mascara_borde.shape, 3), dtype=np.uint8)
    mv_borde[mascara_borde]  = [255, 140, 0]
    mv_borde[~mascara_borde] = [15,  18,  35]
    axes[1].imshow(mv_borde)
    axes[1].set_title(
        f"SAM 2 — borde tolva  (score: {score_sam:.2f})\n"
        f"Naranja=borde metálico  |  Negro=material",
        color=T, fontsize=9)
    axes[1].axis("off")

    # Panel 3: máscara solo material
    mv_mat = np.zeros((*mascara_material.shape, 3), dtype=np.uint8)
    mv_mat[mascara_material]  = [0, 195, 255]
    mv_mat[~mascara_material] = [15, 18, 35]
    axes[2].imshow(mv_mat)
    axes[2].set_title(
        "Máscara — solo material interior\n(invertida + recorte asimétrico)",
        color=T, fontsize=9)
    axes[2].text(0.5, 0.03,
                  f"Material: {mascara_material.mean()*100:.1f}%  |  "
                  f"{mascara_material.sum():,} px",
                  transform=axes[2].transAxes, ha="center",
                  color=L, fontsize=8)
    axes[2].axis("off")

    # Panel 4: overlay final
    ov = img_np.copy().astype(np.float32)
    ov[mascara_material] = ov[mascara_material]*0.25 \
                           + np.array([0, 229, 255])*0.75
    cont = cv2.dilate(
        mascara_material.astype(np.uint8), np.ones((4,4), np.uint8)) \
        - mascara_material.astype(np.uint8)
    ov[cont.astype(bool)] = [255, 220, 0]
    axes[3].imshow(ov.astype(np.uint8))
    axes[3].set_title("Overlay — material segmentado", color=T, fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Figura: {ruta_salida}")


# ── Pipeline principal ────────────────────────────────────

def detectar_tolva(ruta_imagen, prompt=None):
    if prompt is None:
        prompt = PROMPT_ACTIVO

    print(f"\nProcesando: {ruta_imagen}")
    img_bgr = cv2.imread(str(ruta_imagen))
    if img_bgr is None:
        print(f"ERROR: no se pudo cargar {ruta_imagen}")
        return None, None, None

    img_np  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_np)
    h, w    = img_np.shape[:2]
    print(f"  Resolución: {w}×{h} px")

    processor, model_dino = cargar_grounding_dino()
    predictor_sam         = cargar_sam2()

    # Detección bbox
    print(f"\nDetectando con prompt: '{prompt}'")
    bbox, score_bbox = detectar_bbox(img_pil, processor, model_dino, prompt)

    if bbox is None:
        print("  ⚠ No detectado — probando prompts alternativos...")
        for p_alt in PROMPTS_ALTERNATIVOS:
            if p_alt == prompt:
                continue
            print(f"  → '{p_alt}'")
            bbox, score_bbox = detectar_bbox(
                img_pil, processor, model_dino, p_alt,
                box_threshold=0.20)
            if bbox is not None:
                print(f"  ✓ Detectado con '{p_alt}'")
                break

    if bbox is None:
        print("\n✗ No se detectó la tolva.")
        return None, None, img_np

    print(f"✓ Bbox: {tuple(bbox)}  score={score_bbox:.3f}")

    # PASO A: SAM 2 → borde de la tolva
    print("\n[A] SAM 2 segmentando borde de la tolva...")
    mascara_borde, score_sam = segmentar_borde_tolva(
        img_np, bbox, predictor_sam)

    # PASO B+C: invertir + recorte asimétrico → solo material
    print("\n[B+C] Invirtiendo + recorte asimétrico...")
    mascara_material, region = extraer_material(
        mascara_borde, bbox, img_np.shape)

    # Visualizar y guardar
    ruta_fig = DIR_RESULTADOS / f"resultado_02_{Path(ruta_imagen).stem}.png"
    visualizar(img_np, bbox, mascara_borde, mascara_material,
               score_bbox, score_sam, region, ruta_fig)

    np.save("mascara_tolva.npy", mascara_material)
    np.save("bbox_tolva.npy", np.array(bbox))
    print(f"\n✓ Guardado: mascara_tolva.npy  bbox_tolva.npy")
    print(f"\nPróximo paso:")
    print(f"  python3 03_estimar_volumen.py {ruta_imagen} \\")
    print(f"          --vacia imagenes_prueba/tolva_vacia.png")

    return mascara_material, bbox, img_np


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta_default = Path("imagenes_prueba/tolva_llena.png")
        if ruta_default.exists():
            ruta = str(ruta_default)
            print(f"Usando imagen por defecto: {ruta}")
        else:
            print("Uso: python3 02_detectar_tolva.py <ruta_imagen>")
            sys.exit(1)

    mascara, bbox, imagen = detectar_tolva(ruta)

    if mascara is not None:
        ys, xs = np.where(mascara)
        print(f"\n{'='*52}")
        print(f"  Bbox:      {tuple(bbox)}")
        print(f"  Material:  {mascara.sum():,} px  "
              f"({mascara.mean()*100:.1f}%)")
        if len(xs) > 0:
            print(f"  Rango X:   [{xs.min()}, {xs.max()}]")
            print(f"  Rango Y:   [{ys.min()}, {ys.max()}]")
        print(f"")
        print(f"  Ajuste fino:")
        print(f"    CORTE_CABINA_PCT    = {CORTE_CABINA_PCT} "
              f"(sube si entra la cabina)")
        print(f"    MARGEN_SUPERIOR_PCT = {MARGEN_SUPERIOR_PCT} "
              f"(sube si entra borde superior)")
        print(f"    MARGEN_INFERIOR_PCT = {MARGEN_INFERIOR_PCT} "
              f"(sube si entra borde inferior)")
        print(f"    KERNEL_CLOSING      = {KERNEL_CLOSING} "
              f"(baja si conecta bordes, sube para rellenar más)")
        print(f"{'='*52}")
