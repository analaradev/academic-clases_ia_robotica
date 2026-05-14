import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
MODEL_PATH = r"../Models/face_landmarker.task"
SIGNATURE_FILE = "face_signature.csv"
# THRESHOLD: The lower this number, the stricter the match.
# 0.02 is strict, 0.05 is loose. Tune this!
MATCH_THRESHOLD = 0.03

# --- LOAD SIGNATURE ---
try:
    saved_signature = np.loadtxt(SIGNATURE_FILE, delimiter=",")
    # Reshape back to (478, 2) so we can do math easily
    saved_signature = saved_signature.reshape(-1, 2)
    print("Signature loaded successfully.")
except Exception as e:
    print(f"Error loading signature: {e}")
    exit()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LATEST_RESULT = None


def save_result(result, output_image, timestamp_ms):
    global LATEST_RESULT
    LATEST_RESULT = result


# Helper to calculate difference between two faces
def calculate_similarity(live_landmarks, saved_data):
    # Convert live landmarks to numpy array
    live_data = np.array([[lm.x, lm.y] for lm in live_landmarks])

    # Simple Logic: Calculate average distance between all corresponding points
    # We subtract the center of the face to handle "position" differences (centering)
    # But for a simple class demo, we can just compare the raw shapes relative to the bounding box

    # Calculate the error (Mean Squared Error)
    error = np.mean((live_data - saved_data) ** 2)
    return error


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=save_result)

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if LATEST_RESULT and LATEST_RESULT.face_landmarks:
            # Check every detected face
            for i, landmarks in enumerate(LATEST_RESULT.face_landmarks):

                # Compare live face to saved face
                error_score = calculate_similarity(landmarks, saved_signature)

                # Determine Identity
                if error_score < MATCH_THRESHOLD:
                    color = (0, 255, 0)  # Green
                    text = f"ACCESS GRANTED (Err: {error_score:.4f})"
                else:
                    color = (0, 0, 255)  # Red
                    text = f"UNKNOWN FACE (Err: {error_score:.4f})"

                # Visualization
                # Calculate bounding box for drawing text
                x_vals = [lm.x for lm in landmarks]
                y_vals = [lm.y for lm in landmarks]
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)

                h, w, _ = frame.shape
                cv2.rectangle(frame, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), color, 2)
                cv2.putText(frame, text, (int(x_min * w), int(y_min * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Face Recognition System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()