# YOLO 與 MediaPipe 在跌倒偵測中的應用

本系統利用 YOLO 與 MediaPipe 來進行人體檢測與關鍵點提取，並根據人體幾何關係計算跌倒分數（fall_score）。
當分數超過預設閾值時，系統認定檢測到跌倒事件。

---

## 技術概述

### YOLO 物件偵測

- **作用：**
  - 利用 YOLO v8 模型檢測影像中是否存在「person」，並鎖定目標區域。
  - 檢測結果會回傳各類物件的標籤與邊界框資訊。

- **使用方式：**
  - 輸入完整影像，模型返回所有檢測到的物件及其邊界框（例如：x1, y1, x2, y2）。
  - 在跌倒偵測中，當判定目標為「person」時，從該區域裁切影像以進行後續的關鍵點提取。

### MediaPipe Pose 姿勢估算

- **作用：**
  - 在 YOLO 檢測到的「person」區域內使用 MediaPipe Pose 提取人體的主要關鍵點（landmarks）。

- **使用方式：**
  - 對裁切後的人物圖像進行顏色空間轉換，輸入 MediaPipe 模型。
  - 取得的 landmarks 包含頭部、肩膀、臀部、腳踝、腿部等部位的 (x, y) 位置及可信度（visibility）。
  - 為避免單幀檢測不準確，系統使用滑動窗口平滑 (Sliding Window Average) 技術對關鍵點位置進行平滑處理。

---

## 跌倒分數計算

### 基本邏輯

根據提取到的平滑後的 landmarks，系統主要從三個面向計算跌倒分數：

1. **頭部與腳踝高度差 (score_head)**
2. **軀幹傾斜角 (score_torso)**
3. **腿部角度 (score_leg)**

每個部分經過線性轉換後，分別乘以權重，最後加權平均得到整體跌倒分數。

### 詳細計算步驟

#### 1. 頭部與腳踝高度差 (score_head)
- **資料來源：**
  - 使用 `landmarks[0].y` 表示頭部位置，使用 `landmarks[27].y` 與 `landmarks[28].y`（左右腳踝）計算其平均值。
- **計算方式：**
  - 計算高度差：
    `head_ankle_diff = ((landmarks[27].y + landmarks[28].y) / 2) - landmarks[0].y`
  - 若高度差小於一定值（經過扣除0.1），使用線性比例轉換，並由 `1.0 -` 此比例得到得分：
    `score_head = 1.0 - clamp((head_ankle_diff - 0.1) / 0.4, 0.0, 1.0)`
  - 當人物趴下或跌倒時，頭部與腳踝的高度差會顯著減少，使得該分數增高。

#### 2. 軀幹傾斜角 (score_torso)
- **資料來源：**
  - 使用左右肩膀（landmarks 11 與 12）計算肩部中心，
  - 使用左右臀部（landmarks 23 與 24）計算臀部中心。
- **計算方式：**
  - 計算肩部與臀部中心之間的水平 (dx) 與垂直 (dy) 位移，進而計算出與垂直方向夾角（用 `angle_from_vertical(dx, dy)` 函數返回角度值）。
  - 定義傾斜角：
    - 當角度 ≤ 30° 時，score_torso 為 0（代表較為直立）。
    - 當角度 ≥ 90° 時，score_torso 為 1（代表明顯倒下）。
    - 介於兩者則進行線性插值：
      `score_torso = (deg_torso - 30) / 60.0`

#### 3. 腿部角度 (score_leg)
- **資料來源：**
  - 分別計算左腿（landmarks[25] 與 landmarks[23]）與右腿（landmarks[26] 與 landmarks[24]）與垂直方向的角度。
- **計算方式：**
  - 使用 `angle_from_vertical(dx, dy)` 計算左右腿與垂直方向的角度，取較大值作為代表。
  - 當角度 ≤ 30° 時，score_leg 為 0；當角度 ≥ 90° 時，score_leg 為 1；介於之間則進行線性插值：
    `score_leg = (deg_leg - 30) / 60.0`

#### 4. 綜合跌倒分數 (fall_score)
- **加權平均公式：**
  - 使用權重分別為 0.4（頭部）、0.4（軀幹）與 0.2（腿部）。
  - 最終計算公式：
    `fall_score = 0.4 * score_head + 0.4 * score_torso + 0.2 * score_leg`
  - 當 fall_score 超過預設閾值（如 0.5），則系統將該情況視為跌倒事件。

---

## 小結

本系統結合 YOLO 與 MediaPipe 優勢，先透過 YOLO 精準鎖定人物區域，再利用 MediaPipe 提取關鍵點，並基於人體幾何特性進行跌倒分數計算。
該方法能夠較穩健地檢測出跌倒情形，適用於即時監控與安全應用場景。

