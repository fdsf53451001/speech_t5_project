# Speech T5 project
text to speech (TTS) using T5 model, you can provide wav files or youtube link with specify time to change the pitch of the output sound.

## Installation
```
pip install -r requirements.txt
```

## Use with Youtube video
1. edit video_url, start_time, duration, output_voice_sentence in run.py
2. python run.py

## Manual Usage
1. put your wav files in the `dataset/input_voice/wav` folder
2. run prep_cmu_arctic_spkemb.py file, this will generate speaker embedding.
```
python prep_cmu_arctic_spkemb.py \
        -i dataset/ \
        -o dataset/speaker_embedding/ \
        --splits input_voice \
        -s speechbrain/spkrec-xvect-voxceleb
```
3. run speech_t5_transform.ipynb, change the embedding npy file path first.
```
embedding_np = np.load('PutYourPathHere')
speaker_embeddings = torch.tensor(embedding_np).unsqueeze(0)
```
4. the output wav files will be in the `dataset/output_voice` folder