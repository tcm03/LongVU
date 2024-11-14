import os
import pandas as pd
from yt_dlp import YoutubeDL

# Load the filtered dataset
input_file = 'EnTube_filtered.csv'  # Replace with your actual file path
output_base_folder = 'EnTube'  # Base folder for videos

# Create subfolders for each label (0, 1, 2)
for label in ['0', '1', '2']:
    os.makedirs(os.path.join(output_base_folder, label), exist_ok=True)

# Load the filtered dataset
df = pd.read_csv(input_file)

# Configure yt-dlp options
def get_ydl_opts(output_folder):
    return {
        'outtmpl': os.path.join(output_folder, '%(id)s.%(ext)s'),  # Save video with ID as filename
        'format': 'best',  # Download the best available quality
        'quiet': False,  # Show download progress
        'ignoreerrors': True,  # Continue even if a video download fails
    }

# Download videos and place them in corresponding label folders
with YoutubeDL() as ydl:
    for _, row in df.iterrows():
        video_url = row['video_link']
        label = str(row['engagement_rate_label'])  # Convert label to string to match folder names
        output_folder = os.path.join(output_base_folder, label)
        
        ydl_opts = get_ydl_opts(output_folder)
        with YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except Exception as e:
                print(f"Failed to download {video_url}: {e}")

print(f"All videos have been downloaded and categorized into folders 0, 1, and 2.")
