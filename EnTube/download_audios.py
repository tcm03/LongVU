import os
import json
from multiprocessing import Pool
import yt_dlp as youtube_dl
import subprocess
import shutil
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--json_dir', type=str, help="folder contains list json file <video-id>.json", required=True)
parser.add_argument('--nj', type=int, default=4)
parser.add_argument('--audio_dir', type=str, required=True)
# parser.add_argument('--server_url', type=str, default="")
args = parser.parse_args()

# list_videos = args.json_dir
njobs = args.nj
audio_dir = args.audio_dir

os.makedirs(audio_dir, exist_ok=True)

def post_process(d):
    global audio_dir
    if d['status'] == 'finished' and d['postprocessor'] == 'ExtractAudio':
        idx = d.get('info_dict').get('id')
        label = d.get('info_dict').get('label')
        if not os.path.isabs(audio_dir):
            audio_dir = os.path.join(os.getcwd(), audio_dir)
        audio_path = os.path.join(audio_dir, str(label), idx + '.wav')
        tmp_audio_path = os.path.join(audio_dir, str(label), idx + '-tmp.wav')
        # print(f'audio_path: {audio_path}')
        if os.path.isfile(audio_path):
            # Convert the audio file using ffmpeg
            subprocess.run(
                ['ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', tmp_audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # print(f'FILEPATH_TMP: {tmp_audio_path}, CHECK: {os.path.isfile(tmp_audio_path)}')
            # print(f'FILEPATH: {audio_path}, CHECK: {os.path.isfile(audio_path)}')
            # Replace the original file with the converted file
            shutil.move(tmp_audio_path, audio_path)  # Use shutil to move the file

def download(vid):
    id, label = vid
    # Ensure the label-specific directory exists
    os.makedirs(os.path.join(audio_dir, str(label)), exist_ok=True)

    audio_path = os.path.join(audio_dir, str(label), id + '.wav')
    if os.path.exists(audio_path):
        return

    ytb_link = "https://www.youtube.com/watch?v=" + id
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': os.path.join(audio_dir, str(label), '%(id)s.%(ext)s'),
        'postprocessor_hooks': [post_process],
        'merge_output_format': 'wav',  # To ensure consistent extension
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.params['label'] = label
        ydl.download([ytb_link])


EnTube_df = pd.read_csv('EnTube_filtered.csv')
for _, row in EnTube_df.iterrows():
    video_id = row['video_id']
    label = row['engagement_rate_label']
    download((video_id, label))


# with Pool(njobs) as p:
#     p.map(download, total_samples)