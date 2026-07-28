# Object Recognition using OpenCV (YOLOv4-tiny)

Real-time object detection using OpenCV's DNN module and the pre-trained **YOLOv4-tiny** model. Detects and labels **80 object classes** (COCO dataset) in images, videos, or a live webcam feed.

## Example

| Input | Output |
|---|---|
| ![input](examples/input_dog.jpg) | ![output](examples/output_dog.jpg) |

Detected: **dog (87%)**, **truck (81.5%)**, **bicycle (60.6%)**

## Project Structure

```
object-recognition-opencv/
├── object_recognition.py   # Main script
├── download_models.sh      # Downloads model files
├── requirements.txt
├── models/                 # Created after running download_models.sh
├── test_images/            # Sample test image
├── examples/                # Before/after images for this README
└── output/                  # Annotated results saved here
```

## Setup

```bash
git clone https://github.com/<your-username>/object-recognition-opencv.git
cd object-recognition-opencv

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
bash download_models.sh
```

> **Note:** Use `opencv-python<5.0.0`. Version 5.x removed Darknet model support (`readNetFromDarknet`), which this project relies on.

## Usage

```bash
python object_recognition.py --image test_images/dog.jpg
python object_recognition.py --video path/to/video.mp4
python object_recognition.py --webcam
```

Press `q` to stop a video/webcam stream. Results are saved automatically in `output/`.

## How It Works

- **YOLOv4-tiny** predicts bounding boxes and class probabilities in a single forward pass, making it fast enough for real-time CPU inference.
- OpenCV's `cv2.dnn` loads the model directly from its Darknet `.cfg` and `.weights` files.
- **Non-Maximum Suppression (NMS)** removes duplicate overlapping boxes for the same object.

## Requirements

- Python 3.8+
- opencv-python < 5.0.0
- numpy < 2.0.0

## Task Reference

Built for the OpenCV task: **Object Recognition**, chosen from the list (Face Recognition, Object Tracking, Object Recognition, Line Tracking, Color Recognition, Tag Recognition, Object Classification).
