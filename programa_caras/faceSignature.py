import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
MODEL_PATH = r"../Models/face_landmarker.task"
SAVE_FILE = "face_signature.csv"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable for result
LATEST_RESULT = None


def save_result(result, output_image, timestamp_ms):
    global LATEST_RESULT
    LATEST_RESULT = result


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=save_result)

cap = cv2.VideoCapture(0)

print("Press 's' to SAVE your face signature.")
print("Press 'q' to QUIT.")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if LATEST_RESULT and LATEST_RESULT.face_landmarks:
            # Visual feedback: Draw a green box to show we see a face
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 255, 0), 2)

        cv2.imshow('Registration', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and LATEST_RESULT and LATEST_RESULT.face_landmarks:
            # --- EXTRACT THE SIGNATURE ---
            # We take the 478 landmarks and flatten them into a list of x,y coordinates
            # We use the FIRST detected face [0]
            landmarks = LATEST_RESULT.face_landmarks[0]

            # Create a simple list of 478 pairs of (x, y)
            signature = []
            for lm in landmarks:
                signature.append(lm.x)
                signature.append(lm.y)
                # We ignore Z for simplicity in this demo

            # Save to file
            np.savetxt(SAVE_FILE, signature, delimiter=",")
            print(f"Face signature saved to {SAVE_FILE}!")
            break

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()