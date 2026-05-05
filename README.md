# Cubicacion v2 - deteccion de tolva

Flujo minimo para generar `resultados/resultado_02_tolva_llena.png`.

## Archivos necesarios

- `02_detectar_tolva.py`: detecta y segmenta la tolva.
- `config.py`: define modelos, prompt, thresholds y carpeta de resultados.
- `requirements.txt`: dependencias Python.
- `imagenes_prueba/tolva_llena.png`: imagen de ejemplo para reproducir el resultado.

No se deben subir a GitHub: `venv_tolva/`, `checkpoints/`, `resultados/`, `__pycache__/`, `*.npy` ni `debug_*.png`.

## Instalacion

Python recomendado: 3.10, 3.11 o 3.12.

```bash
python3 -m venv venv_tolva
source venv_tolva/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

En Apple Silicon, PyTorch usa MPS si esta disponible. En NVIDIA, usa CUDA si la instalacion local de PyTorch lo soporta. En CPU funciona, pero sera mas lento.

## Ejecutar

```bash
python3 02_detectar_tolva.py imagenes_prueba/tolva_llena.png
```

Salida principal:

```text
resultados/resultado_02_tolva_llena.png
```

Tambien genera archivos auxiliares locales:

```text
mascara_tolva.npy
bbox_tolva.npy
```

Estos auxiliares estan ignorados por git porque son resultados generados.
