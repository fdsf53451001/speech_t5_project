import subprocess
from yt_to_wav import yt_download_wav
from speech_t5_transform import generate_voice

# config
video_url = 'https://www.youtube.com/watch?v=QWDngzLPX5E'
start_time = 6711  # in seconds
duration = 1    # in seconds
yt_wav_filename = 'dataset/input_voice/wav/'+'yt_audio.wav'
output_voice_sentence = 'hello world'


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