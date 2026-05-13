import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math

# --- CONFIGURATION ---
# Use your specific model path here
MODEL_PATH = r"C:\Users\luiso\PycharmProjects\AutomatizacionRobotica\hand_landmarker.task"

# --- CONSTANTS ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to hold the latest result from the async callback
LATEST_RESULT = None


def is_okay_sign(hand_landmarks):
    # Extract coordinates for readability
    # Note: These are normalized (0.0 to 1.0), so distance is unitless
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    # Logic 1: Check if Thumb and Index are touching
    # We use Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
    distance_thumb_index = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )

    # Threshold: < 0.05 is usually a good "touching" distance in normalized coordinates
    is_touching = distance_thumb_index < 0.05

    # Logic 2: Check if other fingers are extended
    # A simple automation check: Is the tip higher (lower y value) than the base of the finger?
    # Or simply: Is the tip far from the wrist?
    # Let's use a simple Y-check (assuming hand is held upright):
    # Tip (12) should be higher (smaller y) than its PIP joint (10)
    middle_extended = middle_tip.y < hand_landmarks[10].y
    ring_extended = ring_tip.y < hand_landmarks[14].y
    pinky_extended = pinky_tip.y < hand_landmarks[18].y

    others_extended = middle_extended and ring_extended and pinky_extended

    return is_touching and others_extended

# --- HELPER: DRAWING FUNCTION ---
# The new API doesn't have a direct drawing utility, so we create a simple one using OpenCV
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks

    # Create a copy of the image to draw on
    annotated_image = rgb_image.copy()
    height, width, _ = annotated_image.shape

    # Loop through the detected hands
    for hand_landmarks in hand_landmarks_list:

        # 1. Draw Keypoints (Dots)
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

        # 2. Draw Connections (Lines) - Optional manually defining standard connections
        # Define standard hand connections (indices of landmarks)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
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


# --- STEP 2: CALLBACK FUNCTION ---
def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global LATEST_RESULT
    LATEST_RESULT = result


# --- STEP 3: INITIALIZE LANDMARKER ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=save_result)  # Pass the callback here

# --- STEP 4: MAIN LOOP ---
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame received from OpenCV to MediaPipe's Image object
        # Note: OpenCV is BGR, MediaPipe Image needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp (required for LIVE_STREAM mode)
        frame_timestamp_ms = int(time.time() * 1000)

        # Send to MediaPipe (Async)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Check if we have a result to draw
        if LATEST_RESULT:
            # Draw landmarks on the frame
            # Note: We draw on the original BGR frame for display
            frame = draw_landmarks_on_image(frame, LATEST_RESULT)

        if LATEST_RESULT and LATEST_RESULT.hand_landmarks:
            # Get the first hand detected
            hand0_landmarks = LATEST_RESULT.hand_landmarks[0]

            # Draw the landmarks (using our previous function)
            frame = draw_landmarks_on_image(frame, LATEST_RESULT)

            # --- AUTOMATION LOGIC HERE ---
            if is_okay_sign(hand0_landmarks):
                cv2.putText(frame, "STATUS: OK SIGN DETECTED", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "STATUS: WAITING...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Landmarker (New API)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()