import soundfile as sf
import librosa
import numpy as np
from scipy import signal
from IPython.display import Audio
import sys, glob


source_dir = r"translation_audio"
hrtf_dir = r"classify_speech/diffuse"

_source = glob.glob(source_dir)

_source = glob.glob(source_dir)
_hrtf = glob.glob(hrtf_dir)

print("List of Sources:")
for s in range(len(_source)):
    print(_source[s])


# Load the audio source
audio, sr_audio = librosa.load(f"{source_dir}/Recording (3).wav", sr=48000)  # adjust path
if audio.ndim > 1:  # Convert stereo to mono if needed
    audio = audio.mean(axis=1)

# Load the HRTF file
hrtf, sr_hrtf = sf.read(f"{hrtf_dir}/hrtf.wav")  # adjust path

# Resample if sample rates differ
if sr_audio != sr_hrtf:
    audio = librosa.resample(audio, sr_audio, sr_hrtf)
    sr_audio = sr_hrtf

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
    
# Convolve audio with left and right HRTF channels
left_convolved = signal.fftconvolve(audio, hrtf[:, 0], mode='full')
right_convolved = signal.fftconvolve(audio, hrtf[:, 1], mode='full')

# Stack the left and right channels to form a binaural stereo signal
binaural_audio = np.vstack((left_convolved, right_convolved)).T

# Normalize audio to prevent clipping
max_val = np.abs(binaural_audio).max()
if max_val > 0:
    binaural_audio /= max_val


# Save the binaural output to a WAV file
sf.write("output_binaural.wav", binaural_audio, sr_audio)

# Listen to the binaural audio
Audio(binaural_audio, rate=sr_audio)

