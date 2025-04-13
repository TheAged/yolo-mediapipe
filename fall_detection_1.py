import cv2
import mediapipe as mp
import math
from ultralytics import YOLO
import time

# =================================================================
# Section 1: Basic Parameter Settings
# =================================================================
YOLO_MODEL_PATH = "/home/tku-im-sd/backend_project/yolov8n.pt"
FALL_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.55

# =================================================================
# Section 2: Initialize YOLO Model and MediaPipe Pose
# =================================================================
def load_yolo_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        raise

yolo_model = load_yolo_model(YOLO_MODEL_PATH)

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# =================================================================
# Section 3: Sliding Window Smoothing for Landmarks
# =================================================================
WINDOW_SIZE = 5
landmark_history = {}

class SmoothedLandmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility

def smooth_landmarks_window(landmarks):
    global landmark_history
    smoothed = []
    # Process each landmark with a sliding window average
    for i, lm in enumerate(landmarks):
        if i not in landmark_history:
            landmark_history[i] = []
        if lm.visibility >= VISIBILITY_THRESHOLD:
            landmark_history[i].append((lm.x, lm.y))
        if len(landmark_history[i]) > WINDOW_SIZE:
            landmark_history[i].pop(0)
        if len(landmark_history[i]) > 0:
            avg_x = sum(x for x, y in landmark_history[i]) / len(landmark_history[i])
            avg_y = sum(y for x, y in landmark_history[i]) / len(landmark_history[i])
            smoothed.append(SmoothedLandmark(avg_x, avg_y, lm.visibility))
        else:
            smoothed.append(SmoothedLandmark(lm.x, lm.y, lm.visibility))
    return smoothed

# =================================================================
# Section 4: Fall Score Calculation
# =================================================================
def angle_from_vertical(dx, dy):
    if abs(dy) < 1e-6:
        return 90.0
    rad = math.atan(abs(dx) / abs(dy))
    return math.degrees(rad)

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def compute_fall_score(landmarks):
    # (A) Height difference between head and ankles using landmark 0 and landmarks 27, 28
    head_y = landmarks[0].y
    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
    head_ankle_diff = ankle_y - head_y
    if head_ankle_diff < 0:
        head_ankle_diff = 0
    score_head = 1.0 - clamp((head_ankle_diff - 0.1) / 0.4, 0.0, 1.0)
    print(f"[DEBUG] head_ankle_diff: {head_ankle_diff:.3f}, score_head: {score_head:.3f}")

    # (B) Torso inclination using landmarks 11, 12 (shoulders) and 23, 24 (hips)
    shoulder_center_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_center_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_center_y = (landmarks[23].y + landmarks[24].y) / 2
    dx_torso = hip_center_x - shoulder_center_x
    dy_torso = hip_center_y - shoulder_center_y
    deg_torso = angle_from_vertical(dx_torso, dy_torso)
    if deg_torso <= 30:
        score_torso = 0.0
    elif deg_torso >= 90:
        score_torso = 1.0
    else:
        score_torso = (deg_torso - 30) / 60.0
    print(f"[DEBUG] deg_torso: {deg_torso:.1f}, score_torso: {score_torso:.3f}")

    # (C) Leg angle using landmarks for thighs (25, 26) relative to hips (23, 24)
    left_leg_dx = landmarks[25].x - landmarks[23].x
    left_leg_dy = landmarks[25].y - landmarks[23].y
    right_leg_dx = landmarks[26].x - landmarks[24].x
    right_leg_dy = landmarks[26].y - landmarks[24].y
    deg_left_leg = angle_from_vertical(left_leg_dx, left_leg_dy)
    deg_right_leg = angle_from_vertical(right_leg_dx, right_leg_dy)
    deg_leg = max(deg_left_leg, deg_right_leg)
    if deg_leg <= 30:
        score_leg = 0.0
    elif deg_leg >= 90:
        score_leg = 1.0
    else:
        score_leg = (deg_leg - 30) / 60.0
    print(f"[DEBUG] deg_leg: {deg_leg:.1f}, score_leg: {score_leg:.3f}")

    # Weighted average of the three scores
    w_head = 0.4
    w_torso = 0.4
    w_leg = 0.2
    fall_score = w_head * score_head + w_torso * score_torso + w_leg * score_leg
    print(f"[DEBUG] fall_score: {fall_score:.3f}")
    return fall_score

# =================================================================
# Section 5: Process Frame for Fall Detection
# =================================================================
previous_smoothed_landmarks = None

def process_frame(frame):
    """
    Applies YOLO detection on the input BGR frame and uses MediaPipe Pose to extract landmarks for the "person" region.
    The function applies sliding window smoothing for a fall score calculation.
    If the current frame's detection fails, it falls back to previous frame data.
    Returns a tuple: (fall_detected_overall, annotated_frame)
    """
    global previous_smoothed_landmarks
    results = yolo_model.predict(source=frame, device='cpu')
    annotated_frame = results[0].plot(line_width=2)
    fall_detected_overall = False

    print(f"[DEBUG] YOLO results: {results[0].names}")

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names.get(cls, str(cls)) if hasattr(result.names, "get") else result.names[cls]
            if label.lower() == "person" or cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                results_pose = pose_detector.process(person_rgb)
                if not results_pose.pose_landmarks:
                    print("[DEBUG] No Pose detected in current frame; using previous frame data")
                    if previous_smoothed_landmarks is not None:
                        smoothed_landmarks = previous_smoothed_landmarks
                    else:
                        continue
                else:
                    raw_landmarks = results_pose.pose_landmarks.landmark
                    smoothed_landmarks = smooth_landmarks_window(raw_landmarks)
                    previous_smoothed_landmarks = smoothed_landmarks

                fall_score = compute_fall_score(smoothed_landmarks)
                color = (0, 0, 255) if fall_score >= FALL_THRESHOLD else (0, 255, 0)
                text = f"Fall Score: {fall_score:.2f}"
                cv2.putText(annotated_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                if fall_score >= FALL_THRESHOLD:
                    fall_detected_overall = True

    return fall_detected_overall, annotated_frame

# =================================================================
# Section 6: Main Loop for Real-Time Fall Detection Testing
# =================================================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        fall_detected, annotated_frame = process_frame(frame)
        cv2.imshow("Fall Detection", annotated_frame)
        if fall_detected:
            print("[INFO] Fall or abnormal movement detected!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
