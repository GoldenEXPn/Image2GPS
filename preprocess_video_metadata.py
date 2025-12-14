import os
import re
import cv2
import glob
import pandas as pd
import subprocess
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

# Configuration
VIDEO_FOLDER = 'raw_videos'  # Folder containing your .MOV files
OUTPUT_IMAGE_FOLDER = 'dataset/images'
CSV_OUTPUT_PATH = 'dataset/metadata.csv'
FRAMES_PER_SECOND = 4  # Sample rate


os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
def get_gps_from_video(video_path):
    """
    Extracts GPS coordinates (Lat, Lon) from video metadata using ffprobe.
    Returns (None, None) if not found.
    """
    try:
        # Run ffprobe to get metadata in JSON format
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        # Metadata can be in 'format' tags or 'stream' tags
        tags = data.get('format', {}).get('tags', {})
        # Common keys for GPS location in QuickTime/MOV
        location_string = tags.get('location') or \
                          tags.get('com.apple.quicktime.location.ISO6709') or \
                          tags.get('location-eng')
        if location_string:
            # Parse ISO 6709 format (e.g., "+39.9527-075.1920/")
            match = re.search(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', location_string)  # Regex searches for signed floating point numbers
            if match:
                lat = float(match.group(1))
                lon = float(match.group(2))
                return lat, lon
    except Exception as e:
        print(f"Error extracting metadata from {video_path}: {e}")
    return None, None


def get_video_rotation(video_path):
    """Get rotation metadata from video"""
    try:
        cmd = ['ffprobe', '-v', 'quiet',
               '-print_format', 'json',
               '-show_streams',
               video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                rotation = stream.get('tags', {}).get('rotate', '0')
                return int(rotation)
    except:
        pass
    return 0


def rotate_frame(frame, rotation):
    """Rotate frame based on metadata"""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def process_dataset():
    # Get all video files
    video_files = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.MOV")))
    dataset_records = []
    print(f"Found {len(video_files)} videos. Starting extraction...")

    for video_path in video_files:
        video_name = os.path.basename(video_path)

        # 1. Extract GPS from Video Header
        lat, lon = get_gps_from_video(video_path)

        if lat is None or lon is None:
            print(f"WARNING: No GPS found for {video_name}. Skipping...")
            continue

        print(f"Processing {video_name} | GPS: {lat}, {lon}")

        # 2. Extract Frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle corner case
        if fps <= 0: fps = 30
        frame_interval = int(fps / FRAMES_PER_SECOND)

        rotation = get_video_rotation(video_path)  # Get rotation
        frame_count = 0
        saved_count = 0
        video_id = video_name.split('.')[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Apply rotation
                frame = rotate_frame(frame, rotation)
                # Filename format: IMG_08XX_frame001.jpg
                image_filename = f"{video_id}_frame{saved_count:03d}.jpg"
                image_full_path = os.path.join(OUTPUT_IMAGE_FOLDER, image_filename)

                cv2.imwrite(image_full_path, frame)

                # Append to records
                dataset_records.append({
                    "file_name": f'images/{image_filename}',
                    "latitude": lat,
                    "longitude": lon
                })
                saved_count += 1
            frame_count += 1
        cap.release()

    # 3. Save to CSV
    if dataset_records:
        df = pd.DataFrame(dataset_records)
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"\nSuccess! Processed {len(video_files)} videos.")
        print(f"Generated {len(df)} images.")
        print(f"Metadata saved to: {CSV_OUTPUT_PATH}")
    else:
        print("No records generated. Check if ffmpeg is installed and videos have metadata.")

def load_hf_token(token_path: str = "huggingface_token") -> str:
    token = Path(token_path).read_text().strip()
    if not token:
        raise ValueError("Hugging Face token file is empty.")
    return token


if __name__ == "__main__":
    process_dataset()

    # Upload dataset to huggingface
    hf_token = load_hf_token("huggingface_token")
    login(token=hf_token)
    dataset = load_dataset("imagefolder", data_dir="dataset")
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    dataset.push_to_hub("tianyi-in-the-bush/penncampus_image2gps")

    # Access data
    #print(dataset['train'][0]['image'])  # PIL Image
    #print(dataset['train'][0]['latitude'])  # GPS Label

