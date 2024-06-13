import os
import requests
import yt_dlp
import ffmpeg

# Downloading the CSV file
def download_csv(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)
    print(f"CSV file downloaded: {destination}")

csv_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
csv_path = "balanced_train_segments.csv"

download_csv(csv_url, csv_path)

# Reading the CSV file
def read_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

csv_data = read_csv(csv_path)

# Creating the dataset directory
def create_dataset_directory(path='dataset'):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

dataset_directory = create_dataset_directory()

# Cleaning up the labels
def sanitize_label(label):
    label = label.replace('"', '')
    label = label.replace('/', '.')
    return label

# Generating the output filename
def get_output_filename(video_id, labels):
    sanitized_labels = [sanitize_label(label) for label in labels]
    filename = f"{video_id}_{'-'.join(sanitized_labels)}.mp3"
    return filename

# Downloading the audio segment
def download_audio_segment(video_id, start_time, end_time, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'verbose': False
    }
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_url = info_dict['url']
            # Edit the audio file within the specified time range using ffmpeg
            (
                ffmpeg
                .input(video_url, ss=start_time, t=str(float(end_time) - float(start_time)))
                .output(output_path, format='mp3', acodec='libmp3lame')
                .run(quiet=True, overwrite_output=True)
            )
        print(f"Downloaded: {video_id}")
    except Exception as e:
        pass  # Ignore errors

# Main processing loop
for line in csv_data:
    if line.startswith('#'):
        continue  # Skip comment lines
    
    fields = line.strip().split(',')
    video_id = fields[0]
    start_time = fields[1]
    end_time = fields[2]
    labels = fields[3:]
    
    output_filename = get_output_filename(video_id, labels)
    output_path = os.path.join(dataset_directory, output_filename)
    
    download_audio_segment(video_id, start_time, end_time, output_path)
