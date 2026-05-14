# Programa de Clasificacion de Imagenes con Deep Learning

## Descripcion
Este programa entrena un modelo de clasificacion de imagenes usando TensorFlow/Keras con transfer learning (MobileNetV2). El modelo fue entrenado para reconocer dos objetos: **botellas** y **tazas**.

Todo el codigo esta contenido en un solo archivo: `clasificador.py`.

## Objetos clasificados

| Clase | Descripcion |
|---|---|
| **botellas** | Botellas de plastico o vidrio |
| **tazas** | Tazas o mugs |

## Dataset

Se usaron **60 imagenes reales** (30 por clase) extraidas del ejemplo del curso. Las imagenes tienen variabilidad en:
- Posicion del objeto
- Fondo diferente
- Iluminacion
- Formatos (jpg y png)

Las imagenes se encuentran en `dataset/botellas/` y `dataset/tazas/`.

## Archivos

| Archivo | Descripcion |
|---|---|
| `clasificador.py` | **Codigo principal unificado**. Contiene entrenamiento, prediccion por webcam y prediccion de imagen. |
| `modelo_clasificacion.keras` | Modelo entrenado (9.2 MB). |
| `clases.txt` | Lista de clases detectadas: botellas, tazas. |
| `grafica_entrenamiento.png` | Grafica de precision y perdida durante el entrenamiento. |
| `dataset/` | Carpeta con 30 imagenes de botellas y 30 de tazas. |

## Resultados del entrenamiento

- **Epochs:** 15
- **Precision de entrenamiento:** 100%
- **Precision de validacion:** 91.67%
- **Perdida de validacion:** 0.0686

## Requisitos

- Python 3.11
- Webcam (solo si usas el comando `webcam`)

## Como ejecutar

### Paso 1: Activar entorno virtual

```bash
cd /Users/usuario/Desktop/ClasesOlmos/programa_clasificacion_imagenes
source venv_clasif/bin/activate
```

### Paso 2: Entrenar el modelo

```bash
python clasificador.py entrenar
```

Esto lee las imagenes de `dataset/`, entrena el modelo y guarda los archivos:
- `modelo_clasificacion.keras`
- `clases.txt`
- `grafica_entrenamiento.png`

### Paso 3: Probar con la webcam

```bash
python clasificador.py webcam
```

Muestra un objeto a la camara. El programa mostrara si detecta "BOTELLAS" o "TAZAS" con el porcentaje de confianza. Presiona **'q'** para salir.

### Paso 3 alternativo: Probar con una imagen

```bash
python clasificador.py imagen "dataset/botellas/botella 1.jpg"
```

## Capturas de pantalla para la tarea

1. **Terminal del entrenamiento**: Captura mostrando las 15 epochs y la precision final (91.67%).
2. **Grafica**: La imagen `grafica_entrenamiento.png` que se genero automaticamente.
3. **Prediccion webcam**: Foto de la ventana clasificando un objeto.
4. **Dataset**: Captura mostrando las carpetas `dataset/botellas/` y `dataset/tazas/` con las imagenes.

## Notas

- El modelo usa MobileNetV2 como base pre-entrenada, lo cual hace que el entrenamiento sea rapido incluso en CPU.
- Las imagenes reales del ejemplo dieron mejor precision (91.67%) que las sinteticas.
- Todo el codigo esta en un solo archivo (`clasificador.py`) para facilitar la entrega y ejecucion.
