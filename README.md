# Speech T5 project
text to speech (TTS) using T5 model, you can provide wav files or youtube link with specify time to change the pitch of the output sound.

## Installation
```
pip install -r requirements.txt
```

## Way1. Input sentence with Youtube video
1. edit video_url, start_time, duration, output_voice_sentence in run.py
2. python run.py

## Way2. Talk to ChatGPT with voice output
1. you must run run.py first to extract the speaker embedding
2. Setup OPENAI_API_KEY, npy file path in ConversationSystem.py
2. python ConversationSystem.py

### Notice
- When the agent reply pop out, you need to wait for it to generate voice. This will take about 20 seconds.
- Comment out the "generate_voice" fuction in line 49 to stop voice generation in conversation.

## Way3. Manual Usage
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