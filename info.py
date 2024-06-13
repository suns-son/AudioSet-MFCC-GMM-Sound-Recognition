import os
import json
from collections import defaultdict

# Load translated_ontology.json
with open('translated_ontology.json', 'r', encoding='utf-8') as f:
    translated_ontology = json.load(f)

# Load original ontology.json
with open('ontology.json', 'r', encoding='utf-8') as f:
    original_ontology = json.load(f)

# Create dictionaries to map label IDs to their names in English and Turkish
label_map_tr = {entry['id']: entry['name'] for entry in translated_ontology}
label_map_en = {entry['id']: entry['name'] for entry in original_ontology}

# Function to categorize videos by labels
def categorize_videos(dataset_path):
    video_files = [f for f in os.listdir(dataset_path) if f.endswith('.mp3')]
    total_videos = len(video_files)
    label_counts = defaultdict(int)

    for video in video_files:
        # Extract the part of the filename between the first and last dot
        start = video.find('.')
        end = video.rfind('.')
        if start != -1 and end != -1 and start < end:
            label_part = video[start:end]
            # Replace dots with slashes
            label_part = label_part.replace('.', '/')
            # Split by dashes to get labels
            labels = label_part.split('-')
            for label in labels:
                label_id = label.strip()
                label_counts[label_id] += 1

    return total_videos, label_counts

# Path to the dataset directory
dataset_path = 'dataset'
total_videos, label_counts = categorize_videos(dataset_path)

# Sort the labels by count (descending) and alphabetically for ties
sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], label_map_tr.get(x[0], '')))

# Print total number of videos
print(f"Total number of videos: {total_videos}")

# Print number of videos per label with label names in both Turkish and English
print("\nNumber of videos per label:")
for label_id, count in sorted_labels:
    label_name_tr = label_map_tr.get(label_id, 'Unknown Label')
    label_name_en = label_map_en.get(label_id, 'Unknown Label')
    print(f"({label_id}) - {label_name_tr} ({label_name_en}): {count}")
