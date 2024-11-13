import pandas as pd
from pytube import YouTube
from tqdm import tqdm

# Read the dataset
input_file = "EnTube.csv"  # Replace with your actual file path
output_file = "EnTube_filtered.csv"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(input_file)

def is_video_accessible(video_url):
    """Check if the YouTube video is accessible."""
    try:
        YouTube(video_url).check_availability()
        return True
    except:
        return False

# Use tqdm to show progress bar during the filtering process
tqdm.pandas(desc="Filtering videos")
df['is_accessible'] = df['video_link'].progress_apply(is_video_accessible)
filtered_df = df[df['is_accessible'] == True].drop(columns=['is_accessible'])

# Write the filtered data to a new CSV file
filtered_df.to_csv(output_file, index=False)

# Print the number of filtered records
print(f"Number of accessible videos: {len(filtered_df)}")
print(f"Total number of videos: {len(df)}")