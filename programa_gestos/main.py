import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math

# --- CONFIGURACION ---
# Ruta local del modelo (mismo archivo que se uso en clase)
MODEL_PATH = "ejemplo/hand_landmarker.task"

# --- CONSTANTES ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Variable global para guardar el ultimo resultado del callback async
LATEST_RESULT = None


# =========================
# FUNCION: DETECTAR GESTO PULGAR ARRIBA
# =========================
# Modificacion respecto al codigo base: en lugar de detectar el gesto OK,
# aqui se detecta si el pulgar esta extendido hacia arriba y los demas
# dedos estan doblados.
def is_thumbs_up(hand_landmarks):
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    # distancia de cada dedo a la muneca
    dist_thumb = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    dist_index = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
    dist_middle = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
    dist_ring = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y)
    dist_pinky = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y)

    # umbrales ajustados para deteccion comoda
    thumb_threshold = 0.28
    fingers_down_threshold = 0.24

    # pulgar extendido y arriba
    thumb_extended = dist_thumb > thumb_threshold
    thumb_above = thumb_tip.y < wrist.y

    # demas dedos doblados
    index_down = dist_index < fingers_down_threshold
    middle_down = dist_middle < fingers_down_threshold
    ring_down = dist_ring < fingers_down_threshold
    pinky_down = dist_pinky < fingers_down_threshold

    return thumb_extended and thumb_above and index_down and middle_down and ring_down and pinky_down


# =========================
# FUNCION: DETECTAR GESTO VICTORIA
# =========================
# Dos dedos extendidos (indice y medio), los demas doblados.
def is_victory_sign(hand_landmarks):
    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    dist_index = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
    dist_middle = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
    dist_ring = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y)
    dist_pinky = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y)

    finger_up = 0.30
    finger_down = 0.22

    index_up = dist_index > finger_up
    middle_up = dist_middle > finger_up
    ring_down = dist_ring < finger_down
    pinky_down = dist_pinky < finger_down

    return index_up and middle_up and ring_down and pinky_down


# =========================
# FUNCION: DETECTAR GESTO L
# =========================
# Pulgar extendido horizontalmente e indice extendido verticalmente.
def is_l_sign(hand_landmarks):
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    dist_thumb = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    dist_index = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
    dist_middle = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
    dist_ring = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y)
    dist_pinky = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y)

    thumb_threshold = 0.28
    index_threshold = 0.30
    fingers_down_threshold = 0.24

    thumb_extended = dist_thumb > thumb_threshold
    index_extended = dist_index > index_threshold
    middle_down = dist_middle < fingers_down_threshold
    ring_down = dist_ring < fingers_down_threshold
    pinky_down = dist_pinky < fingers_down_threshold

    return thumb_extended and index_extended and middle_down and ring_down and pinky_down


# =========================
# FUNCION: DETECTAR CORAZON COREANO 
# =========================
# Pulgar e indice se tocan formando un corazon, los demas dedos doblados.
def is_korean_heart(hand_landmarks):
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    # distancia entre punta del pulgar y punta del indice
    dist_thumb_index = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    # distancia de los demas dedos a la muneca
    dist_middle = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
    dist_ring = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y)
    dist_pinky = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y)

    # umbrales
    heart_threshold = 0.12  # pulgar e indice muy cerca
    fingers_down_threshold = 0.24

    # pulgar e indice casi tocandose
    fingers_touching = dist_thumb_index < heart_threshold

    # demas dedos doblados
    middle_down = dist_middle < fingers_down_threshold
    ring_down = dist_ring < fingers_down_threshold
    pinky_down = dist_pinky < fingers_down_threshold

    return fingers_touching and middle_down and ring_down and pinky_down


# --- FUNCION AUXILIAR: DIBUJAR LANDMARKS ---
# La nueva API no trae utilidad de dibujo directa, asi que se hace una
# funcion propia con OpenCV para mostrar los puntos y conexiones.
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = rgb_image.copy()
    height, width, _ = annotated_image.shape

    for hand_landmarks in hand_landmarks_list:
        # 1. Dibujar puntos clave
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

        # 2. Dibujar conexiones
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),  # Indice
            (5, 9), (9, 10), (10, 11), (11, 12),  # Medio
            (9, 13), (13, 14), (14, 15), (15, 16),  # Anular
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Menique
        ]

        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]

            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)

            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return annotated_image


# --- PASO 2: FUNCION CALLBACK ---
def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global LATEST_RESULT
    LATEST_RESULT = result


# --- PASO 3: INICIALIZAR LANDMARKER ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=save_result
)

# --- PASO 4: LOOP PRINCIPAL ---
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR (OpenCV) a RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calcular timestamp (requerido para modo LIVE_STREAM)
        frame_timestamp_ms = int(time.time() * 1000)

        # Enviar a MediaPipe (async)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if LATEST_RESULT:
            frame = draw_landmarks_on_image(frame, LATEST_RESULT)

        if LATEST_RESULT and LATEST_RESULT.hand_landmarks:
            hand0_landmarks = LATEST_RESULT.hand_landmarks[0]

            # --- LOGICA DE AUTOMATIZACION (GESTOS) ---
            # Se prueban los gestos en orden y se muestra el primero detectado
            if is_thumbs_up(hand0_landmarks):
                cv2.putText(frame, "GESTO: PULGAR ARRIBA", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_victory_sign(hand0_landmarks):
                cv2.putText(frame, "GESTO: VICTORIA", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_l_sign(hand0_landmarks):
                cv2.putText(frame, "GESTO: L", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_korean_heart(hand0_landmarks):
                cv2.putText(frame, "GESTO: CORAZON COREANO", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "ESPERANDO GESTO...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Detector de Gestos - Proyecto', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
