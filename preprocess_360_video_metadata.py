import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# CONFIG
VIDEO_DIR = "D:/360 Video"
OUTPUT_DIR = "D:/dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
GEOJSON_PATH = "track.geojson"

FPS = 1  # frames per second to extract
IMAGE_EXT = ".jpg"

os.makedirs(IMAGE_DIR, exist_ok=True)

do_preprocess = False

if do_preprocess:
    # LOAD GEOJSON TRACK
    with open(GEOJSON_PATH, "r") as f:
        geo = json.load(f)

    # Handle nested coordinate structures safely
    raw_coords = geo["features"][0]["geometry"]["coordinates"]

    # Flatten: [[ [lon,lat,elev,time], ... ]] -> [ [lon,lat,elev,time], ... ]
    if isinstance(raw_coords[0][0], list):
        raw_coords = raw_coords[0]

    track = np.array(
        [[c[0], c[1], c[3]] for c in raw_coords],
        dtype=np.float64
    )

    track_lons = track[:, 0]
    track_lats = track[:, 1]
    track_times = track[:, 2]

    t_start = track_times.min()
    t_end = track_times.max()

    # INTERPOLATION FUNCTION
    def interpolate_latlon(t):
        lon = np.interp(t, track_times, track_lons)
        lat = np.interp(t, track_times, track_lats)
        return lat, lon

    # PROCESS VIDEOS
    dataset_records = []

    video_files = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ])

    for vid_idx, video_name in enumerate(video_files):
        video_path = os.path.join(VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open {video_name}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / video_fps

        frame_interval = int(video_fps / FPS)

        frame_idx = 0
        saved_idx = 0

        pbar = tqdm(total=int(duration * FPS), desc=video_name)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Normalized time in video [0,1]
                t_norm = frame_idx / frame_count
                t_global = t_start + t_norm * (t_end - t_start)

                lat, lon = interpolate_latlon(t_global)

                image_filename = f"video{vid_idx}_{saved_idx:06d}{IMAGE_EXT}"
                image_path = os.path.join(IMAGE_DIR, image_filename)

                cv2.imwrite(image_path, frame)

                dataset_records.append({
                    "file_name": f"images/{image_filename}",
                    "latitude": float(lat),
                    "longitude": float(lon)
                })

                saved_idx += 1
                pbar.update(1)

            frame_idx += 1

        pbar.close()
        cap.release()

    # WRITE METADATA

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    if dataset_records:
        df = pd.DataFrame(dataset_records)
        df.to_csv(metadata_path, index=False)
        print(f"\nSuccess! Processed {len(video_files)} videos.")
        print(f"Generated {len(df)} images.")
        print(f"Metadata saved to: {metadata_path}")
    else:
        print("No records generated. Check if ffmpeg is installed and videos have metadata.")

    print(f"Done. Images: {len(dataset_records)}")
    print(f"Metadata written to {metadata_path}")

# UPLOAD DATA TO HUGGINGFACE

def load_hf_token(token_path: str = "huggingface_token") -> str:
    token = Path(token_path).read_text().strip()
    if not token:
        raise ValueError("Hugging Face token file is empty.")
    return token



# Upload dataset to huggingface
hf_token = load_hf_token("huggingface_token")
login(token=hf_token)


dataset = load_dataset("imagefolder", data_dir=OUTPUT_DIR)

# Split train/test
train_test_ratio = 0.2  # fraction for test set
train_test_split = dataset['train'].train_test_split(
    test_size=train_test_ratio,
    shuffle=True,
    seed=42
)

dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

dataset_dict.push_to_hub(
    "aaron-jiang/penncampus_image2gps",
    commit_message="Add train/test split"
)

print(dataset['train'][0]['image'])  # PIL Image
print(dataset['train'][0]['latitude'])  # GPS Label

