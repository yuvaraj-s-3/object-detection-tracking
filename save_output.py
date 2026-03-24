import cv2
import numpy as np
from ultralytics import YOLO

VIDEO   = "tire1.mp4"
OUTPUT  = "tire_detected_output.mp4"
CONF    = 0.10
FRAME_W = 640
FRAME_H = 360

model = YOLO("yolov8n.pt")
cap   = cv2.VideoCapture(VIDEO)
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (FRAME_W, FRAME_H)
)

tracker         = None
tracking_active = False
tracked_box     = None
lost_count      = 0
frame_count     = 0
prev_gray       = None
REDETECT_EVERY  = 25
MAX_LOST        = 30


def is_valid(x1, y1, x2, y2, frame):
    fh, fw = frame.shape[:2]
    w  = x2 - x1
    h  = y2 - y1
    cy = (y1 + y2) / 2
    if cy < fh * 0.35: return False, 0
    if w > fw * 0.40 or h > fh * 0.40: return False, 0
    if h > 0 and (w / h) > 7.0: return False, 0
    roi = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
    if roi.size == 0: return False, 0
    mean_b = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    if mean_b > 145: return False, 0
    return True, (145 - mean_b) * min(w, h)


def get_motion(frame, prev_gray):
    curr = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if prev_gray is None: return curr, None
    _, motion = cv2.threshold(cv2.absdiff(curr, prev_gray), 8, 255, cv2.THRESH_BINARY)
    return curr, motion


def detect(frame, motion):
    best_box, best_score = None, 0
    for r in model(frame, conf=CONF, verbose=False):
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            valid, score = is_valid(x1, y1, x2, y2, frame)
            if not valid: continue
            if motion is not None:
                fh, fw = frame.shape[:2]
                m = motion[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                if m.size > 0:
                    score *= (1 + cv2.countNonZero(m) / m.size * 2)
            score *= float(box.conf[0])
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2 - x1, y2 - y1)
    return best_box


def draw_box(frame, x, y, w, h, color):
    fh, fw = frame.shape[:2]
    cx   = x + w // 2
    cy   = y + h // 2
    side = max(w, h) + 6
    sx   = max(0,  cx - side // 2)
    sy   = max(0,  cy - side // 2)
    ex   = min(fw, sx + side)
    ey   = min(fh, sy + side)
    cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)


def show_status(frame, text, color):
    cv2.putText(frame, text, (8, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


print(f"Processing {total} frames... please wait")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    prev_gray, motion = get_motion(frame, prev_gray)

    need_detect = (
        not tracking_active
        or lost_count > MAX_LOST
        or frame_count % REDETECT_EVERY == 0
    )

    if need_detect:
        box = detect(frame, motion)
        if box is not None:
            tracker         = cv2.TrackerCSRT_create()
            tracker.init(frame, box)
            tracked_box     = box
            tracking_active = True
            lost_count      = 0
        else:
            if tracking_active: lost_count += 1
            if lost_count > MAX_LOST: tracking_active = False

    elif tracking_active and tracker is not None:
        ok, box = tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in box]
            if 2 < w < FRAME_W * 0.5 and 2 < h < FRAME_H * 0.5:
                tracked_box = (x, y, w, h)
                lost_count  = 0
            else:
                tracking_active = False
                lost_count      = MAX_LOST + 1
        else:
            lost_count += 1
            if lost_count > MAX_LOST: tracking_active = False

    if tracked_box is not None and lost_count == 0:
        draw_box(frame, *tracked_box, color=(0, 255, 0))
        show_status(frame, "Detecting", (0, 255, 0))
    else:
        show_status(frame, "Searching...", (0, 165, 255))

    writer.write(frame)

    if frame_count % 100 == 0:
        pct = int(frame_count / total * 100)
        print(f"  {pct}% done ({frame_count}/{total})")

cap.release()
writer.release()
print(f"\nDone! Output saved: {OUTPUT}")
