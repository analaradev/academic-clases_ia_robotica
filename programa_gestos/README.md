# Programa de Deteccion de Gestos Personalizados

## Descripcion
Este programa detecta gestos de la mano en tiempo real usando MediaPipe y OpenCV. Fue desarrollado como parte de la tarea de Vision Artificial, modificando la base proporcionada en clase (`visionArtificial.py`).

## Gestos implementados
Se programo el reconocimiento de cuatro gestos distintos:

1. **Pulgar arriba** 👍 — pulgar extendido hacia arriba, demas dedos doblados.
2. **Victoria** ✌️ — indice y medio extendidos, demas dedos doblados.
3. **L** 🤙 — pulgar e indice extendidos formando una L, demas dedos doblados.
4. **Corazon coreano** 🫰 — pulgar e indice juntos formando un corazon, demas dedos doblados.

## Archivos

| Archivo | Descripcion |
|---|---|
| `detector_gestos.py` | Codigo principal basado en MediaPipe (modificacion de visionArtificial.py) |
| `visionArtificial.py` | Codigo base proporcionado por el docente |
| `hand_landmarker.task` | Modelo de MediaPipe para deteccion de manos |
| `captura_pulgar_arriba.png` | Captura del sistema reconociendo "Pulgar arriba" |
| `captura_victoria.png` | Captura del sistema reconociendo "Victoria" |
| `captura_l.png` | Captura del sistema reconociendo "L" |

## Requisitos

- Python 3.10 o 3.11 (MediaPipe no es compatible con Python 3.14)
- Webcam funcional

### Librerias necesarias
```
pip install mediapipe==0.10.18 opencv-python==4.10.0.84
```

## Instrucciones de uso

1. Abre una terminal en esta carpeta.
2. Instala las dependencias:
   ```bash
   pip install mediapipe==0.10.18 opencv-python==4.10.0.84
   ```
3. Ejecuta el programa:
   ```bash
   python detector_gestos.py
   ```
4. Aparecera la ventana de la camara con los landmarks dibujados.
5. Realiza alguno de los tres gestos frente a la camara.
6. El programa mostrara el nombre del gesto detectado en la esquina superior.
7. Presiona **'q'** para salir.

## Diferencias respecto al codigo base

- **Nuevos gestos:** En lugar de detectar solo el gesto "OK", se implementaron cuatro gestos diferentes (Pulgar arriba, Victoria, L, Corazon coreano).
- **Logica de deteccion:** Se utilizo la distancia euclidiana de cada punta de dedo a la muneca para determinar si un dedo esta extendido o doblado, lo cual permite una deteccion mas comoda y robusta.
- **Una sola mano:** Se limito la deteccion a una mano (`num_hands=1`) para mayor estabilidad.
- **Comentarios en espanol:** Todo el codigo esta comentado en espanol para facilitar la comprension.

## Notas importantes

- Asegurate de tener buena iluminacion para que MediaPipe detecte bien la mano.
- Si la deteccion no es precisa, puedes ajustar los umbrales en las funciones `is_thumbs_up`, `is_victory_sign` e `is_l_sign`.
- El modelo `hand_landmarker.task` debe estar en la misma carpeta que el script.
