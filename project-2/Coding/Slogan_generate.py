import os
import pandas as pd
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import subprocess
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import random
import warnings
from transformers.utils import logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# === CONFIG ===
# === Path setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# font path
FONT_PATH_TITLE = os.path.join(BASE_DIR, "BowlbyOne-Regular.ttf")
FONT_PATH_SUB = "/Library/Fonts/Arial.ttf"  

# Excel path
PC_EXCEL_PATH = os.path.join(BASE_DIR, "Final_PC_Users_Travel_Ads_Updated.xlsx")
MOBILE_EXCEL_PATH = os.path.join(BASE_DIR, "Final_Mobile_Users_Travel_Ads_Prompts.xlsx")

# image folder
PC_IMAGE_FOLDER = os.path.join(BASE_DIR, "PCoutput")
MOBILE_IMAGE_FOLDER = os.path.join(BASE_DIR, "Mobileoutput")

# output path
FINAL_OUTPUT_PC = os.path.join(BASE_DIR, "final_outputs/pc")
FINAL_OUTPUT_MB = os.path.join(BASE_DIR, "final_outputs/mobile")

os.makedirs(FINAL_OUTPUT_PC, exist_ok=True)
os.makedirs(FINAL_OUTPUT_MB, exist_ok=True)

FRAME_COUNT = 20
FINAL_HOLD_FRAMES = 30

DEVICE = "cpu"

# === Generate advertising slogans ===
def generate_slogan_from_prompts(user_prompt, image_prompt):
    prompt = f"""You are a poetic travel copywriter, skilled in writing compelling ad slogans for social media and digital campaigns.

Given the user's personal style and the image concept, write one short, catchy, and poetic travel slogan (\u226415 words).

The slogan should be emotionally appealing, personalized, and evoke a desire to explore. 

User description: {user_prompt}
Image concept: {image_prompt}

Only return the slogan. No explanation, no extra lines."""
    result = subprocess.run(["ollama", "run", "llama3"], input=prompt.encode(), capture_output=True)
    raw_output = result.stdout.decode("utf-8").strip()
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    return lines[-1].strip('"\u201c\u201d\u2018\u2019\' .\u3002\uff01!?') if lines else "[No slogan generated]"

# === Text line feed ===
def wrap_text(text, font, draw, max_width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = current + " " + word if current else word
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

# === Detection of the key area ===
def detect_main_objects(image_path, threshold=0.8):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    results = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=threshold)[0]
    return [box.cpu().tolist() for box in results["boxes"]]

# === Determine the text placement area ===
def choose_text_position(image_size, boxes, text_block_size):
    width, height = image_size
    block_w, block_h = text_block_size
    candidates = {
        "bottom_left": (50, height - block_h - 50),
        "bottom_right": (width - block_w - 50, height - block_h - 50),
        "top_left": (50, 50),
        "top_right": (width - block_w - 50, 50)
    }
    for name, (x, y) in candidates.items():
        block = [x, y, x + block_w, y + block_h]
        overlap = False
        for box in boxes:
            if not (box[2] < block[0] or box[0] > block[2] or box[3] < block[1] or box[1] > block[3]):
                overlap = True
                break
        if not overlap:
            return (x, y)
    return (50, height - block_h - 50)

# === Composite animation ===
def overlay_text_gif(image_path, title, subtitle, output_path):
    base = Image.open(image_path).convert("RGBA")
    width, height = base.size

    # Randomize font sizes
    title_size = max(20, int(width * random.uniform(0.05, 0.07)))
    subtitle_size = max(10, int(width * random.uniform(0.035, 0.045)))

    font_title = ImageFont.truetype(FONT_PATH_TITLE, title_size)
    font_sub = ImageFont.truetype(FONT_PATH_SUB, subtitle_size)

    # Wrap subtitle into multiple lines;
    max_width = int(width * random.uniform(0.75, 0.95))
    lines = wrap_text(subtitle, font_sub, ImageDraw.Draw(base), max_width)
    total_text_height = title_size + len(lines) * (subtitle_size + 10)

    # Detect safe text position (away from detected objects)
    boxes = detect_main_objects(image_path)
    pos_x, pos_y = choose_text_position((width, height), boxes, (max_width, total_text_height))

    # Randomly choose animation effect
    effect = random.choice(["fade", "blur", "scale", "wave", "typewriter"])
    print(f"🎞 Using effect: {effect}")

    frames = []
    for i in range(FRAME_COUNT):
        progress = i / (FRAME_COUNT - 1)
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Font scaling (for 'scale' effect)
        if effect == "scale":
            scale = 0.5 + 0.5 * progress
            font_title_temp = ImageFont.truetype(FONT_PATH_TITLE, int(title_size * scale))
            font_sub_temp = ImageFont.truetype(FONT_PATH_SUB, int(subtitle_size * scale))
        else:
            font_title_temp = font_title
            font_sub_temp = font_sub

        # Vertical offset for sliding effects
        offset = int((1 - progress) * 30) if effect in ["slide", "scale"] else 0
        tx, ty = pos_x, pos_y - offset

        alpha = int(progress * 255)

        # Draw title with shadow
        draw.text((tx + 2, ty + 2), title.upper(), font=font_title_temp, fill=(100, 100, 100, alpha))
        draw.text((tx, ty), title.upper(), font=font_title_temp, fill=(255, 255, 255, alpha))

        # Draw subtitle
        sub_y = ty + font_title_temp.size + 10
        for idx, line in enumerate(lines):
            # typewriter: show part of the line by progress
            if effect == "typewriter":
                chars = int(len(line) * progress)
                line_to_draw = line[:chars]
            else:
                line_to_draw = line

            # wave: add vertical sinusoidal bounce
            wave_offset = int(10 * math.sin(progress * math.pi * 2 + idx)) if effect == "wave" else 0
            line_y = sub_y - offset + wave_offset

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((pos_x + dx, line_y + dy), line_to_draw, font=font_sub_temp, fill=(0, 0, 50, int(alpha * 0.5)))

            # main text
            draw.text((pos_x, line_y), line_to_draw, font=font_sub_temp, fill=(255, 255, 255, alpha))

            sub_y += font_sub_temp.size + 10

            #draw.text((pos_x + 1, line_y + 1), line_to_draw, font=font_sub_temp, fill=(0, 0, 100, alpha))
            #draw.text((pos_x, line_y), line_to_draw, font=font_sub_temp, fill=(255, 255, 255, alpha))
            #sub_y += font_sub_temp.size + 10

        # Optional blur
        if effect == "blur":
            layer = layer.filter(ImageFilter.GaussianBlur(radius=(1 - progress) * 5))

        # Merge text layer with base image
        composed = Image.alpha_composite(base, layer)
        frames.append(composed.convert("RGB"))

    # Hold final frame
    frames.extend([frames[-1]] * FINAL_HOLD_FRAMES)

    # Save animated GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

# === Batch execution ===
def process_ads_from_excel(excel_path, image_folder, image_prefix, output_folder, output_prefix):
    df = pd.read_excel(excel_path)
    for i, row in df.iterrows():
        no = int(row["No."])
        user_prompt = str(row["user prompt"]).strip()
        image_prompt = str(row["image_prompt"]).strip()
        location = str(row["Location"]).strip()

        # name method
        image_filename = f"{image_prefix}{no}.png"
        image_path = os.path.join(image_folder, image_filename)

        output_filename = f"{output_prefix}_ad_{no}.gif"
        output_path = os.path.join(output_folder, output_filename)

        print(f"\n📸 Processing {image_filename} ({location})")
        slogan = generate_slogan_from_prompts(user_prompt, image_prompt)
        print(f"📝 Slogan: {slogan}")

        overlay_text_gif(image_path, location, slogan, output_path)
        print(f"✅ Saved to {output_path}")


# === MAIN ===
if __name__ == "__main__":
    process_ads_from_excel(
        PC_EXCEL_PATH,
        PC_IMAGE_FOLDER,
        image_prefix="PCoutput_",
        output_folder=FINAL_OUTPUT_PC,
        output_prefix="pc"
    )
    
    process_ads_from_excel(
        MOBILE_EXCEL_PATH,
        MOBILE_IMAGE_FOLDER,
        image_prefix="Mobileoutput_",
        output_folder=FINAL_OUTPUT_MB,
        output_prefix="mb"
    )

