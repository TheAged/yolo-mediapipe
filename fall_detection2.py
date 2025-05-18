import cv2
import mediapipe as mp
import math
from ultralytics import YOLO
import time

# ==========================
# 基本參數設定
# ==========================
# 使用 COCO 預訓練權重 (請將 yolov8n.pt 放置在正確位置)
YOLO_MODEL_PATH = "/home/tku-im-sd/backend_project/yolov8n.pt"
# 當落地分數 (fall_score) 大於此值時，認定為跌倒
FALL_THRESHOLD = 0.5  
# 若關鍵點 visibility 小於此值，不更新滑動窗口；使用之前數據作備援
VISIBILITY_THRESHOLD = 0.55  

# ==========================
# 初始化 YOLO 與 MediaPipe Pose
# ==========================
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

# ==========================
# 滑動窗口平滑 (Sliding Window Average)
# ==========================
WINDOW_SIZE = 5
# 保存各關鍵點的歷史資料，會持久保存，不在每幀中重置
landmark_history = {}

class SmoothedLandmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility

def smooth_landmarks_window(landmarks):
    global landmark_history
    smoothed = []
    # 為每個 landmark 進行滑動窗口平均處理
    for i, lm in enumerate(landmarks):
        if i not in landmark_history:
            landmark_history[i] = []
        if lm.visibility >= VISIBILITY_THRESHOLD:
            landmark_history[i].append((lm.x, lm.y))
        # 保持窗口大小
        if len(landmark_history[i]) > WINDOW_SIZE:
            landmark_history[i].pop(0)
        # 若有歷史數據則計算平均，不然直接使用當前值
        if len(landmark_history[i]) > 0:
            avg_x = sum(x for x, y in landmark_history[i]) / len(landmark_history[i])
            avg_y = sum(y for x, y in landmark_history[i]) / len(landmark_history[i])
            smoothed.append(SmoothedLandmark(avg_x, avg_y, lm.visibility))
        else:
            smoothed.append(SmoothedLandmark(lm.x, lm.y, lm.visibility))
    return smoothed

# ==========================
# 落地分數計算
# ==========================
def angle_from_vertical(dx, dy):
    if abs(dy) < 1e-6:
        return 90.0
    rad = math.atan(abs(dx) / abs(dy))
    return math.degrees(rad)

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def compute_fall_score(landmarks):
    # (A) 頭部與腳踝高度差：取 landmark 0 與左右腳踝 (27,28)
    head_y = landmarks[0].y
    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
    head_ankle_diff = ankle_y - head_y
    if head_ankle_diff < 0:
        head_ankle_diff = 0
    score_head = 1.0 - clamp((head_ankle_diff - 0.1) / 0.4, 0.0, 1.0)
    # Debug
    print(f"[DEBUG] head_ankle_diff: {head_ankle_diff:.3f}, score_head: {score_head:.3f}")
    
    # (B) 躯幹傾斜角：利用左右肩 (11,12) 與左右臀 (23,24)
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
    # Debug
    print(f"[DEBUG] deg_torso: {deg_torso:.1f}, score_torso: {score_torso:.3f}")

    # (C) 腿部角度：以左右大腿 (25,26) 與垂直方向角度衡量
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
    # Debug
    print(f"[DEBUG] deg_leg: {deg_leg:.1f}, score_leg: {score_leg:.3f}")

    # 加權平均
    w_head = 0.4
    w_torso = 0.4
    w_leg = 0.2
    fall_score = w_head * score_head + w_torso * score_torso + w_leg * score_leg
    # Debug
    print(f"[DEBUG] fall_score: {fall_score:.3f}")
    return fall_score

# 用來保存上一幀的平滑 Landmark 資料，作為備援
previous_smoothed_landmarks = None

def process_frame(frame):
    """
    對輸入的 BGR 影像進行 YOLO 檢測，並對 "person" 區域使用 MediaPipe Pose 提取關鍵點，
    以滑動窗口平滑方式計算落地分數 (fall_score)。
    若本幀檢測失敗則使用上一幀資料備援。
    回傳 (fall_detected_overall, annotated_frame)
    """
    global previous_smoothed_landmarks
    results = yolo_model.predict(source=frame, device='cpu')
    annotated_frame = results[0].plot(line_width=2)
    fall_detected_overall = False

    # Debug: 記錄 YOLO 檢測結果
    print(f"[DEBUG] YOLO 檢測結果: {results[0].names}")
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            # 取得標籤 (確保兼容字典或列表)
            label = result.names.get(cls, str(cls)) if hasattr(result.names, "get") else result.names[cls]
            if label.lower() == "person" or cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                results_pose = pose_detector.process(person_rgb)
                if not results_pose.pose_landmarks:
                    print("[DEBUG] 當前幀未偵測到 Pose，使用前幀備援")
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

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 無法開啟攝影機")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 讀取影像失敗")
            break
        fall_detected, annotated_frame = process_frame(frame)
        cv2.imshow("Fall Detection", annotated_frame)
        if fall_detected:
            print("[INFO] 檢測到跌倒或異常動作！")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
