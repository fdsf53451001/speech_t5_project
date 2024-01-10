import subprocess
from yt_to_wav import yt_download_wav
from speech_t5_transform import generate_voice

# config
video_url = 'https://www.youtube.com/watch?v=iPP9AtTbhC4'
start_time = 76  # in seconds
duration = 4    # in seconds
yt_wav_filename = 'dataset/input_voice/wav/'+'yt_audio.wav'
output_voice_sentence = "Vedal is the creator of all AI's in the Neuroverse—Both named and unnamed AI were programmed and trained by him. He develops and maintains Neuro, and sometimes joins her on streams"


# download
yt_download_wav(video_url, start_time, duration, yt_wav_filename)

# extract speaker embedding
subprocess.run([
    'python', 'prep_cmu_arctic_spkemb.py',
    '-i', 'dataset/', 
    '-o', 'dataset/speaker_embedding/', 
    '--splits', 'input_voice',
    '-s', 'speechbrain/spkrec-xvect-voxceleb'
])

# generate audio
generate_voice(output_voice_sentence, 'dataset/speaker_embedding/input_voice-wav-yt_audio.npy', 'dataset/output_voice/yt_audio.wav')