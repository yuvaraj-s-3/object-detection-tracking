# object-detection-tracking
Real-time object detection and tracking using YOLOv8 and OpenCV
# 🎯 Object Detection & Tracking System

A real-time object detection and tracking system built with **YOLOv8** and **OpenCV**.  
Detects and tracks a moving object across all angles, sizes, and lighting conditions — with no GPU required.

---

## 📽️ Demo

The system successfully tracks a tire rolling across a desert sand dune filmed from a drone.  
It handles the object appearing as a **circle**, **oval**, or **thin stick** depending on the camera angle,  
and automatically re-detects it when it re-enters the frame.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Programming language |
| YOLOv8 (Ultralytics) | Object detection |
| OpenCV CSRT Tracker | Smooth real-time tracking |
| NumPy | Frame processing & filtering |

---

## ⚙️ How It Works

1. **Detection** — YOLOv8 scans the frame every 25 frames to detect the object accurately
2. **Tracking** — OpenCV CSRT Tracker handles smooth tracking between detections (faster than running YOLO every frame)
3. **Motion Analysis** — Frame differencing boosts score for moving regions, reducing false detections
4. **Smart Filtering** — Rejects sky, mountains, shadows, and large background objects using:
   - Brightness check (object must be darker than background)
   - Size check (rejects anything too large)
   - Shape check (rejects wide flat shadows)
   - Zone check (ignores top 35% of frame = sky/mountain area)
5. **Auto Re-detection** — When the object leaves the frame, the system waits and automatically re-detects it when it comes back
6. **Output Export** — Processed video with detection boxes is saved as an `.mp4` file

---

## 📁 Project Structure

```
object-detection-tracking/
│
├── main.py              # Live detection and tracking (real-time window)
├── save_output.py       # Processes video and saves output .mp4
├── yolov8n.pt           # YOLOv8 nano pretrained model
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yuvaraj-s-3/object-detection-tracking.git
```

### 2. Install dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 3. Add your video
Place your video file in the project folder and update this line in `main.py`:
```python
VIDEO = "your_video.mp4"
```

### 4. Run live detection
```bash
python main.py
```

### 5. Save output video
```bash
python save_output.py
```
Output is saved as `tire_detected_output.mp4`

---

## 🎨 Visual Output

| Status | Box Color | Message |
|---|---|---|
| Object detected | 🟢 Green | `Detecting` |
| Object temporarily hidden | 🟠 Orange | `Searching...` |
| Object out of frame | 🟠 Orange | `Searching...` |

---

## 💡 Key Design Decisions

- **YOLOv8 Nano** was chosen over larger models to ensure fast performance on CPU
- **CSRT Tracker** was chosen over KCF because it handles scale changes and rotation better
- **Re-detection every 25 frames** balances accuracy and speed — tracker alone drifts over time
- **No training required** — the system uses smart filtering on top of pretrained YOLO weights

---

## 📋 Requirements

```
Python     3.11
ultralytics
opencv-python
numpy
```

---

## 👨‍💻 Author

**Yuvaraj**  
Built as part of a computer vision interview task to detect and track an object in a drone video under challenging conditions including camera movement, changing object angles, and varying object sizes.
