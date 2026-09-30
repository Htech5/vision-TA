"""
Distance estimation via monocular triangle-similarity: distance = (real_width * focal_px) / pixel_width.
Uses the trained model's box width in pixels as the measured width.

Calibrate first (one time, per camera):
    python distance.py --calibrate --class car --known-distance-cm 200 --source 0
    -> stand a known object at a known distance, read the printed focal length,
       put it in FOCAL_LENGTH_PX below.

Then run detection + distance:
    python distance.py --weights runs/human_vehicle/weights/best.pt --source 0
"""
import argparse
from collections import deque
import cv2
from ultralytics import YOLO

STABLE_WINDOW = 15    # frames to average once stable
STABLE_TOLERANCE = 0.03  # max spread allowed (fraction of mean) to call it "stable"

# Average real-world width, in cm, per class. Rough estimates -- for real accuracy,
# calibrate per class with a known-distance shot of that class.
KNOWN_WIDTH_CM = {
    "person": 45,
    "bicycle": 60,
    "car": 180,
    "motorcycle": 80,
    "bus": 250,
    "truck": 250,
}

# Fill this in after running --calibrate once for your camera.
FOCAL_LENGTH_PX = 440.9


def estimate_distance_cm(known_width_cm: float, focal_length_px: float, pixel_width: float) -> float:
    if pixel_width <= 0:
        return float("inf")
    return (known_width_cm * focal_length_px) / pixel_width


def calibrate_focal_length(known_distance_cm: float, known_width_cm: float, pixel_width: float) -> float:
    return (pixel_width * known_distance_cm) / known_width_cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n-seg.pt")
    ap.add_argument("--source", default="0", help="webcam index or video/image path")
    ap.add_argument("--calibrate", action="store_true", help="print focal length instead of distance")
    ap.add_argument("--class", dest="cls_name", default="person", help="class to calibrate against")
    ap.add_argument("--known-distance-cm", type=float, default=200.0)
    ap.add_argument("--show", action="store_true", default=True)
    args = ap.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    readings = deque(maxlen=STABLE_WINDOW)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pixel_width = x2 - x1
            name = model.names[int(box.cls[0])]

            if args.calibrate:
                if name != args.cls_name:
                    continue
                f = calibrate_focal_length(args.known_distance_cm, KNOWN_WIDTH_CM.get(name, 50), pixel_width)
                print(f"{name}: pixel_width={pixel_width:.1f}, focal_length_px={f:.1f}")
                readings.append(f)
                if len(readings) == readings.maxlen:
                    spread = (max(readings) - min(readings)) / (sum(readings) / len(readings))
                    if spread < STABLE_TOLERANCE:
                        avg_f = sum(readings) / len(readings)
                        print(f"\nstable. FOCAL_LENGTH_PX = {avg_f:.1f}")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                continue

            known_w = KNOWN_WIDTH_CM.get(name, 50)
            dist_cm = estimate_distance_cm(known_w, FOCAL_LENGTH_PX, pixel_width)
            label = f"{name} {dist_cm/100:.2f}m"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if args.show:
            cv2.imshow("distance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
