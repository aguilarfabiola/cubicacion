# Cubicacion v2 - deteccion de tolva

Flujo minimo para generar una imagen de diagnostico como:

```text
resultados/resultado_02_tolva_llena.png
```

El script principal es `02_detectar_tolva.py`. Usa Grounding DINO para detectar la tolva/caja del camion y SAM 2 para segmentar la zona. Luego aplica una logica geometrica de recorte para quedarse principalmente con el material dentro de la tolva.

## Archivos necesarios

- `02_detectar_tolva.py`: deteccion, segmentacion y visualizacion.
- `config.py`: modelos, prompt, thresholds y carpeta de resultados.
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

## Alcance actual

Este flujo no esta garantizado para cualquier imagen. Fue ajustado para imagenes parecidas al ejemplo `tolva_llena.png`, donde:

- El camion/tolva aparece claramente visible.
- La tolva se ve principalmente de lado.
- La cabina queda hacia la izquierda.
- El material esta dentro de la tolva y tiene contraste suficiente.
- No hay multiples camiones compitiendo por la deteccion.
- La imagen no esta demasiado borrosa, oscura ni cortada.

Los modelos usados son generales, por lo que pueden detectar otras imagenes. Sin embargo, la etapa posterior de recorte geometrico usa porcentajes fijos y por eso depende bastante del angulo, encuadre y orientacion de la foto.

## Parametros a calibrar

En `config.py`:

- `PROMPT_TOLVA`: texto que usa Grounding DINO para buscar la tolva. Ejemplos: `dump truck hopper`, `dump truck bed`, `truck bed filled with material`.
- `BOX_THRESHOLD`: umbral minimo para aceptar detecciones. Bajarlo puede ayudar si no detecta nada, pero aumenta falsos positivos.
- `TEXT_THRESHOLD`: umbral de coincidencia texto-imagen. Bajarlo puede detectar mas, pero con menor precision.
- `CALIDAD`: selecciona modelos `rapido`, `medio` o `mejor`. En produccion conviene medir precision/tiempo con imagenes reales.

En `02_detectar_tolva.py`:

- `CORTE_CABINA_PCT`: porcentaje del bbox descartado por el lado de la cabina. Si entra cabina en la mascara, subirlo. Si corta material real, bajarlo.
- `MARGEN_SUPERIOR_PCT`: recorte superior del bbox. Si entra borde metalico superior, subirlo. Si corta material, bajarlo.
- `MARGEN_INFERIOR_PCT`: recorte inferior del bbox. Si entra borde inferior/rueda/suelo, subirlo. Si corta material, bajarlo.
- `MARGEN_FRONTAL_PCT`: recorte del extremo frontal opuesto a la cabina. Ajustarlo si entra borde lateral o se pierde material.
- `KERNEL_CLOSING`: tamano del cierre morfologico. Subirlo une huecos pequenos; bajarlo evita unir zonas incorrectas.
- `PROMPTS_ALTERNATIVOS`: lista de prompts usados si falla el prompt principal.

## Consideraciones para produccion

Antes de usarlo en produccion se deberia calibrar con un conjunto de imagenes reales del sitio:

- Probar distintas horas del dia, iluminacion, polvo, lluvia y sombras.
- Probar tolvas llenas, medias, vacias y con distintos materiales.
- Probar varios angulos de camara y distancias.
- Medir falsos positivos, falsos negativos y calidad de mascara.
- Definir una posicion fija de camara si se busca estabilidad.
- Validar que la orientacion esperada se mantenga. Si la cabina puede aparecer a la derecha, hay que adaptar la logica de recorte.
- Guardar ejemplos fallidos para ajustar `PROMPT_TOLVA`, thresholds y porcentajes de recorte.

Para un uso mas robusto, el siguiente paso tecnico seria reemplazar los recortes fijos por una calibracion basada en geometria de la camara, deteccion de orientacion del camion o puntos/poligonos de referencia de la tolva.
