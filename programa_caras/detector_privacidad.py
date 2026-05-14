import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

# --- CONFIGURACION ---
MODEL_PATH = r"Models/blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

LATEST_RESULT = None


# --- FUNCION: DIBUJAR DETECCIONES ---
def draw_face_detections(image, result):
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

        # color azul para rostros detectados
        cv2.rectangle(annotated_image, start_point, end_point, (255, 0, 0), 3)

        # puntos clave (ojos, nariz, boca)
        if detection.keypoints:
            for keypoint in detection.keypoints:
                kx = int(keypoint.x * width)
                ky = int(keypoint.y * height)
                cv2.circle(annotated_image, (kx, ky), 5, (0, 255, 0), -1)

        # puntaje de confianza
        score = detection.categories[0].score
        cv2.putText(annotated_image, f"{score:.2f}",
                    (bbox.origin_x, bbox.origin_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return annotated_image


# --- CALLBACK ---
def save_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global LATEST_RESULT
    LATEST_RESULT = result


# --- INICIALIZACION ---
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_detection_confidence=0.5,
    result_callback=save_result
)

cap = cv2.VideoCapture(0)

print("Detector de Rostros con Privacidad")
print("- 1 rostro  -> Estado normal (verde)")
print("- 0 rostros -> Esperando (naranja)")
print("- 2+ rostros -> PRIVACIDAD ACTIVADA (rojo + blur)")
print("- Presiona 'q' para salir.")
print("")

with FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # voltear para efecto espejo
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        detector.detect_async(mp_image, timestamp_ms)

        # contar rostros detectados
        num_faces = 0
        if LATEST_RESULT and LATEST_RESULT.detections:
            num_faces = len(LATEST_RESULT.detections)

        # =========================
        # LOGICA DE PRIVACIDAD
        # =========================

        if num_faces == 1:
            # estado normal: dibujar rostro y marco verde
            if LATEST_RESULT:
                frame = draw_face_detections(frame, LATEST_RESULT)

            cv2.putText(frame, "1 rostro detectado - OK",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2)

        elif num_faces == 0:
            # no hay nadie
            cv2.putText(frame, "Sin rostros detectados",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 165, 255),
                        2)

        else:
            # 2 o mas rostros: activar privacidad
            # difuminar toda la imagen
            frame = cv2.GaussianBlur(frame, (55, 55), 0)

            # tapar cada rostro con rectangulo negro
            h, w, _ = frame.shape
            if LATEST_RESULT:
                for detection in LATEST_RESULT.detections:
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    bw = bbox.width
                    bh = bbox.height
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 0), -1)

            # mensaje de privacidad
            cv2.putText(frame, "PRIVACIDAD ACTIVADA",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3)
            cv2.putText(frame, f"Demasiados rostros: {num_faces}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

        # contador en esquina inferior
        cv2.putText(frame, f"Rostros: {num_faces}",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        cv2.imshow("Deteccion de Rostros - Privacidad", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")
