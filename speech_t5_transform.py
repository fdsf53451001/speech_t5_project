from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import torch
import numpy as np
from transformers import SpeechT5HifiGan
import soundfile as sf

def generate_voice(sentence, speaker_embedding_path, output_voice_path):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    # text = "Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation"

    inputs = processor(text=sentence, return_tensors="pt")

    embedding_np = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(embedding_np).unsqueeze(0)

    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write(output_voice_path, speech.numpy(), samplerate=16000)