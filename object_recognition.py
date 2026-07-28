"""
Object Recognition using OpenCV + YOLOv4-tiny
-----------------------------------------------
This script detects and recognizes 80 different object classes (from the COCO
dataset) in images, video files, or a live webcam feed using OpenCV's DNN
module and the pre-trained YOLOv4-tiny model.

Usage examples:
    python object_recognition.py --image test_images/sample.jpg
    python object_recognition.py --video test_images/sample.mp4
    python object_recognition.py --webcam
"""

import argparse
import os
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Paths to the model files
# ----------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CFG_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.5   # minimum confidence to keep a detection
NMS_THRESHOLD = 0.4          # non-max suppression threshold (removes overlapping boxes)
INPUT_SIZE = 416             # network input size (width/height)


def load_class_names(path):
    """Read the list of 80 COCO class names, one per line."""
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_network(cfg_path, weights_path):
    """Load the YOLOv4-tiny network with OpenCV's DNN module."""
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # Use CPU by default. If you have CUDA-enabled OpenCV, you can switch to GPU:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_output_layer_names(net):
    """Return the names of the network's output (unconnected) layers."""
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    # OpenCV versions differ: out_layers can be a flat array or array of arrays
    if isinstance(out_layers[0], (list, np.ndarray)):
        return [layer_names[i[0] - 1] for i in out_layers]
    return [layer_names[i - 1] for i in out_layers]


def detect_objects(net, frame, output_layers):
    """Run a forward pass and return the raw network outputs."""
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)
    return net.forward(output_layers)


def postprocess(frame, outputs, class_names):
    """
    Parse network outputs, apply confidence filtering + non-max suppression,
    and draw bounding boxes with class labels on the frame.
    """
    height, width = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detected = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]
            conf = confidences[i]
            detected.append((label, conf))

            color = get_color_for_class(class_ids[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {conf * 100:.1f}%"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 4, y), color, -1)
            cv2.putText(frame, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    return frame, detected


def get_color_for_class(class_id, seed=42):
    """Deterministic, distinct BGR color for each class id."""
    rng = np.random.RandomState(class_id + seed)
    return tuple(int(c) for c in rng.randint(0, 255, size=3))


def process_image(net, output_layers, class_names, image_path, save_dir="output"):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    outputs = detect_objects(net, frame, output_layers)
    frame, detected = postprocess(frame, outputs, class_names)

    print(f"Detected {len(detected)} object(s):")
    for label, conf in detected:
        print(f"  - {label}: {conf * 100:.1f}%")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "result_" + os.path.basename(image_path))
    cv2.imwrite(out_path, frame)
    print(f"Saved annotated image to: {out_path}")

    try:
        cv2.imshow("Object Recognition - Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        # No display available (e.g. running on a headless server) - safe to ignore.
        pass


def process_video(net, output_layers, class_names, source, save_dir="output"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "result_video.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = detect_objects(net, frame, output_layers)
        frame, _ = postprocess(frame, outputs, class_names)

        curr_time = time.time()
        fps_display = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)
        try:
            cv2.imshow("Object Recognition - Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error:
            # No display available (e.g. running on a headless server) - safe to ignore.
            pass

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved annotated video to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Object Recognition using OpenCV + YOLOv4-tiny")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an input image file")
    group.add_argument("--video", type=str, help="Path to an input video file")
    group.add_argument("--webcam", action="store_true", help="Use the live webcam feed")
    args = parser.parse_args()

    for path in (CFG_PATH, WEIGHTS_PATH, NAMES_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing model file: {path}\n"
                f"Run 'bash download_models.sh' first to download the required model files."
            )

    class_names = load_class_names(NAMES_PATH)
    net = load_network(CFG_PATH, WEIGHTS_PATH)
    output_layers = get_output_layer_names(net)

    if args.image:
        process_image(net, output_layers, class_names, args.image)
    elif args.video:
        process_video(net, output_layers, class_names, args.video)
    elif args.webcam:
        process_video(net, output_layers, class_names, 0)


if __name__ == "__main__":
    main()
