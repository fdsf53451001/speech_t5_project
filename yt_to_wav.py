import os
import subprocess
from yt_dlp import YoutubeDL

def yt_download_wav(video_url:str, start_time:int, duration:int=5, output_filename:str='output_audio.wav'):
    # Define options for yt-dlp: download best audio quality
    ydl_opts = {
        'format': 'bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
    }

    # Use yt-dlp to download audio
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Use FFmpeg to crop the audio file
    start = str(start_time)
    duration = str(duration)

    subprocess.run([
        'ffmpeg', '-i', 'downloaded_audio.wav', '-ss', start, '-t', duration,
        '-acodec', 'pcm_s16le', '-ar', '44100', output_filename
    ])

    # Optionally, delete the original downloaded audio
    os.remove('downloaded_audio.wav')

    print(f"Audio saved to {output_filename}")