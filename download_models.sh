#!/usr/bin/env bash
# Downloads the pre-trained YOLOv4-tiny model files required to run
# object_recognition.py. Run this once before using the project:
#
#   bash download_models.sh

set -e

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

echo "Downloading class names (coco.names)..."
curl -L -o "$MODELS_DIR/coco.names" \
  https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

echo "Downloading network configuration (yolov4-tiny.cfg)..."
curl -L -o "$MODELS_DIR/yolov4-tiny.cfg" \
  https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

echo "Downloading pre-trained weights (yolov4-tiny.weights, ~23 MB)..."
curl -L -o "$MODELS_DIR/yolov4-tiny.weights" \
  https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights

echo "Done! Model files saved in: $MODELS_DIR"
