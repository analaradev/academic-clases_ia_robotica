import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

# --- CONFIGURATION ---
MODEL_PATH = r"../Models/blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable for the result
LATEST_RESULT = None


# --- HELPER: DRAWING FUNCTION ---
def draw_face_detections(image, result):
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in result.detections:
        # 1. Draw Bounding Box
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

        # Color: Blue (255, 0, 0) in BGR
        cv2.rectangle(annotated_image, start_point, end_point, (255, 0, 0), 3)

        # 2. Draw Keypoints (Eyes, Nose, Mouth)
        # MediaPipe Face Detector gives 6 keypoints per face
        if detection.keypoints:
            for keypoint in detection.keypoints:
                kx = int(keypoint.x * width)
                ky = int(keypoint.y * height)
                cv2.circle(annotated_image, (kx, ky), 5, (0, 255, 0), -1)

        # 3. Draw Confidence Score
        score = detection.categories[0].score
        cv2.putText(annotated_image, f"{score:.2f}", (bbox.origin_x, bbox.origin_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return annotated_image


# --- CALLBACK ---
def save_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global LATEST_RESULT
    LATEST_RESULT = result


# --- INITIALIZATION ---
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_detection_confidence=0.5,
    result_callback=save_result
)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

with FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp logic
        timestamp_ms = int(time.time() * 1000)

        # Async Detection
        detector.detect_async(mp_image, timestamp_ms)

        # Visualization
        if LATEST_RESULT and LATEST_RESULT.detections:
            frame = draw_face_detections(frame, LATEST_RESULT)

            # --- AUTOMATION LOGIC EXAMPLE ---
            # Example: Trigger if face is "Close" (Bounding box is large)
            first_face = LATEST_RESULT.detections[0]
            face_width = first_face.bounding_box.width

            # If face width > 200 pixels, person is close
            if face_width > 200:
                cv2.putText(frame, "PROXIMITY ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('Face Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()