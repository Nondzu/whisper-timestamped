
# import required modules
import os
import argparse
import sys
from pytube import YouTube
# import whisper
import whisper_timestamped as whisper
import whisper.transcribe 
# from langdetect import detect
import tqdm
import json
import re
import torch

class _CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Set the initial value
    
    def update(self, n=1):  # Ensure n has a default value of 1
        super().update(n)
        self._current += n
        
        # Calculate the percentage progress
        percentage = (self._current / self.total) * 100 if self.total else 0
        
        # Print progress with percentage
        print(f"Progress: {self._current}/{self.total} ({percentage:.2f}%)")

# Inject into tqdm.tqdm of Whisper, so we can see progress
transcribe_module = sys.modules['whisper.transcribe']
transcribe_module.tqdm.tqdm = _CustomProgressBar

def download_audio(yt, output_path, filename):
    """Download the audio if it doesn't exist."""
    full_path = os.path.join(output_path, filename)
    if not os.path.exists(full_path):
        print(f"Downloading audio to {full_path}")
        yt.streams.filter(only_audio=True).first().download(output_path=output_path, filename=filename)
    else:
        print(f"Audio file already exists at {full_path}, skipping download.")

def startfile(path):
    if sys.platform == "win32":
        os.startfile(path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        os.system(opener + " " + path)

def create_and_open_txt(text, filename):
    with open(filename, "w") as file:
        file.write(text)
    startfile(filename)

def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a YouTube video's audio to a text script with language detection.")
    parser.add_argument('url', help="YouTube video URL")
    parser.add_argument('--workdir', default='YoutubeAudios', help="Working directory where all the audio and text files will be saved")
    args = parser.parse_args()

    if not os.path.isdir(args.workdir):
        os.makedirs(args.workdir)
        print(f"Created the directory {args.workdir} for storing the files.")

    yt = YouTube(args.url)

    # Define the output filename based on the video title or ID
    filename = yt.title.replace(" ", "_") + ".mp3"

    # Download audio only if it doesn't exist
    download_audio(yt, args.workdir, filename)

    # Construct the full path to the downloaded file
    audio_file_path = os.path.join(args.workdir, filename)
    audio = whisper.load_audio(audio_file_path)
    # load custom mode
    model_name = "whisper-large-v2-nob.bin"
    hf_state_dict = torch.load(model_name)    # pytorch_model.bin file
    # Rename layers
    for key in list(hf_state_dict.keys())[:]:
        new_key = hf_to_whisper_states(key)
        hf_state_dict[new_key] = hf_state_dict.pop(key)

    # Init Whisper Model and replace model weights
    model = whisper.load_model('large-v2')
    model.load_state_dict(hf_state_dict)

    # TRANCRIPTION
    # model = whisper.load_model()
    result = whisper.transcribe(model, audio, language="pl")

    result = json.dumps(result, indent = 2, ensure_ascii = False)
    print(result)

    # Create a text file name based on the YouTube video and language detected
    text_filename = os.path.splitext(filename)[0] + f"-{model_name}" +".json"
    text_file_path = os.path.join(args.workdir, text_filename)

    # Create and open the text file with the transcribed text
    create_and_open_txt(result, text_file_path)
