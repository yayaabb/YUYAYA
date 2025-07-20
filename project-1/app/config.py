# app/config.py

import os

# COCO dataset paths
COCO_IMAGES_DIR = "coco_dataset/images/val2017"
COCO_ANNOTATIONS_PATH = "coco_dataset/annotations/instances_val2017.json"

# Whisper model setting
WHISPER_MODEL_NAME = "small"

# Score file path
SCORE_FILE = "scores.json"

# Audio recording
AUDIO_FILENAME = "temp_audio.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_DEFAULT = 8