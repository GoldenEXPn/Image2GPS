import json
import cv2
import os
import glob
import pandas as pd

# Config
VIDEO_FOLDER = 'raw_videos'
OUTPUT_IMAGE_FOLDER = 'dataset/images'
JSON_FILE = 'set-0001.json'
FRAMES_PER_SECOND = 0.5  # Sample rate

# Ensure output directory exists
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)


# Helper function
def parse_gps_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    raw_points = data['pointInfoArray'][2]
    gps_list = []
    for point in raw_points:
        parts = point.split(',')
        if len(parts) >= 2:
            gps_list.append({
                'lat': float(parts[0]),
                'lon': float(parts[1])
            })
    return gps_list


'''
Process videos:
1. gps location list
2. Sort videos in ascending order (?)
3. Create csv file, mapping samples from same video to a corresponding gps location
'''
def process_videos():
    # 1. Get GPS Data
    gps_data = parse_gps_data(JSON_FILE)
    # 2. Get Video Files (Descending Order as requested)
    video_files = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.MOV")), reverse=False)
    # Validation
    if len(video_files) != len(gps_data):
        print(f"Warning: Found {len(video_files)} videos but {len(gps_data)} GPS points.")
        # We proceed with the minimum length to avoid crashing
        limit = min(len(video_files), len(gps_data))
        video_files = video_files[:limit]
        gps_data = gps_data[:limit]

    dataset_records = []

    print(f"Starting processing of {len(video_files)} videos...")

    # 3. Iterate through matched Video + GPS pairs
    for idx, (video_path, coords) in enumerate(zip(video_files, gps_data)):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / FRAMES_PER_SECOND) if FRAMES_PER_SECOND else 1

        frame_count = 0
        saved_count = 0
        video_name = os.path.basename(video_path).split('.')[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame based on interval
            if frame_count % frame_interval == 0:
                # Create unique filename: IMG_08XX_001.jpg
                image_filename = f"{video_name}_{saved_count:03d}.jpg"
                image_path = os.path.join(OUTPUT_IMAGE_FOLDER, image_filename)

                cv2.imwrite(image_path, frame)

                # Append to records list
                # Per guidelines: we need Image path, Latitude, Longitude
                dataset_records.append({
                    "file_name": image_filename,  # Relative path preferred for HF
                    "latitude": coords['lat'],
                    "longitude": coords['lon']
                })
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Processed {video_name}: extracted {saved_count} frames.")

    # 4. Save Metadata CSV
    # The guidelines require specific columns
    df = pd.DataFrame(dataset_records)
    csv_path = 'dataset/metadata.csv'
    df.to_csv(csv_path, index=False)
    print(f"Processing complete. Metadata saved to {csv_path}")
    print(f"Total images in dataset: {len(df)}")


if __name__ == "__main__":
    process_videos()