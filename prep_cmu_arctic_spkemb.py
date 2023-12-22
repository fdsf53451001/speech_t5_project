import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

# def f2embed(wav_file, classifier, size_embed):
#     signal, fs = torchaudio.load(wav_file)
    
#     print('original shape', signal.shape)
#     if signal.shape[0] > 1:
#         # only retain one channel, keep shape as [1, T]
#         signal = signal[0, :].unsqueeze(0)

#     signal = signal[:16000]
#     fs = 16000
#     assert fs == 16000, fs
#     with torch.no_grad():
#         embeddings = classifier.encode_batch(signal)
#         embeddings = F.normalize(embeddings, dim=2)
#         embeddings = embeddings.squeeze().cpu().numpy()

#     assert embeddings.shape[0] == size_embed, (embeddings.shape)
#     return embeddings

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)

    # Ensure single-channel input
    if signal.shape[0] > 1:
        signal = signal[0, :].unsqueeze(0)

    # Resample if needed
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
        fs = 16000

    # Assuming 'signal' should have a shape of (1, L) after processing
    assert signal.dim() == 2, f'Signal should be 2-dimensional but got shape {signal.shape}'

    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=1) # Change dim based on actual output
        embeddings = embeddings.squeeze().cpu().numpy()

    # Print out the shape for debugging
    print("Embeddings shape:", embeddings.shape)

    # Now check the embedding size using the correct shape index or logic
    if embeddings.ndim == 1:
        assert len(embeddings) == size_embed, f"Expected embedding size {size_embed}, but got {len(embeddings)}"
    elif embeddings.ndim == 2:
        assert embeddings.shape[1] == size_embed, f"Expected embedding size {size_embed}, but got {embeddings.shape[1]}"
    else:
        raise ValueError(f"Unexpected number of dimensions for embeddings: {embeddings.ndim}")

    return embeddings
def process(args):
    wavlst = []
    for split in args.splits.split(","):
        wav_dir = os.path.join(args.arctic_root, split)
        # print(os.path.join(wav_dir, "wav", "*.wav"))
        wavlst_split = glob.glob(os.path.join(wav_dir, "wav", "*.wav"))
        print(f"{split} {len(wavlst_split)} utterances.")
        wavlst.extend(wavlst_split)

    spkemb_root = args.output_root
    if not os.path.exists(spkemb_root):
        print(f"Create speaker embedding directory: {spkemb_root}")
        os.mkdir(spkemb_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=args.speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[args.speaker_embed]
    for utt_i in tqdm(wavlst, total=len(wavlst), desc="Extract"):
        # TODO rename speaker embedding
        utt_id = "-".join(utt_i.split("/")[-3:]).replace(".wav", "")
        utt_emb = f2embed(utt_i, classifier, size_embed)
        numpy.save(os.path.join(spkemb_root, f"{utt_id}.npy"), utt_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", "-i", required=True, type=str, help="LibriTTS root directory.")
    parser.add_argument("--output-root", "-o", required=True, type=str, help="Output directory.")
    parser.add_argument("--speaker-embed", "-s", type=str, required=True, choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
                        help="Pretrained model for extracting speaker emebdding.")
    parser.add_argument("--splits",  type=str, help="Split of four speakers seperate by comma.",
                        default="cmu_us_bdl_arctic,cmu_us_clb_arctic,cmu_us_rms_arctic,cmu_us_slt_arctic")
    args = parser.parse_args()
    print(f"Loading utterances from {args.arctic_root}/{args.splits}, "
        + f"Save speaker embedding 'npy' to {args.output_root}, "
        + f"Using speaker model {args.speaker_embed} with {spk_model[args.speaker_embed]} size.")
    process(args)

if __name__ == "__main__":
    """
    python prep_cmu_arctic_spkemb.py \
        -i dataset/ \
        -o dataset/speaker_embedding/ \
        --splits input_voice \
        -s speechbrain/spkrec-xvect-voxceleb
    """
    main()