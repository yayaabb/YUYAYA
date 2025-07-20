# app/image_loader.py

import json
import os
from app.config import COCO_IMAGES_DIR, COCO_ANNOTATIONS_PATH

def load_coco_annotations():
    with open(COCO_ANNOTATIONS_PATH, "r") as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    image_objects = {}

    for anno in coco_data["annotations"]:
        image_id = anno["image_id"]
        img_name = f"{image_id:012d}.jpg"
        category_name = categories[anno["category_id"]]

        if img_name not in image_objects:
            image_objects[img_name] = set()
        image_objects[img_name].add(category_name)

    return image_objects