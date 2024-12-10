from yt_dlp import YoutubeDL
import os

EnTube_path = 'D:/EnTube'
EnTube_labels = ['0', '1', '2']
output_base_folder = 'D:/EnTube'

# Configure yt-dlp options
def get_ydl_opts(output_folder):
    return {
        'outtmpl': os.path.join(output_folder, '%(id)s.%(ext)s'),  # Save video with ID as filename
        'format': 'best',  # Download the best available quality
        'quiet': False,  # Show download progress
        'ignoreerrors': True,  # Continue even if a video download fails
    }

def main():
    
    for label in EnTube_labels:
        label_path = os.path.join(EnTube_path, label)
        for video in os.listdir(label_path):
            video_path = os.path.join(label_path, video)
            # if video's path doesn't end in ".mp4"
            if not video.endswith(".mp4"):
                # delete this incomplete video
                os.remove(video_path)
                # extract video id (before ".mp4")
                video_id = video.split(".mp4")[0]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                output_folder = os.path.join(output_base_folder, label)
                ydl_opts = get_ydl_opts(output_folder)
                with YoutubeDL(ydl_opts) as ydl:
                    try:
                        ydl.download([video_url])
                    except Exception as e:
                        print(f"Failed to download {video_url}: {e}")
            


if __name__ == "__main__":
    main()