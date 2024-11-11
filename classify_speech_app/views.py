from django.shortcuts import render
from gtts import gTTS
from googletrans import Translator
import os
from django.conf import settings
import speech_recognition as sr
import base64
from io import BytesIO
import tempfile
from pydub import AudioSegment
import noisereduce as nr
from noisereduce import reduce_noise
import numpy as np
import scipy.io
from scipy.signal import convolve

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format='wav')
    return wav_path



def load_hrtf_filters(hrtf_file_path):
    # Load the HRTF data from the .mat file
    hrtf_data = scipy.io.loadmat(hrtf_file_path)
    
    # Access the HRTF data using the correct keys
    hrtf_left = hrtf_data['hrir_l'].flatten()  # Left ear impulse response
    hrtf_right = hrtf_data['hrir_r'].flatten()  # Right ear impulse response
    
    # Convert float64 to int16
    hrtf_left = (hrtf_left * 32767).astype(np.int16)  # Scale to 16-bit range
    hrtf_right = (hrtf_right * 32767).astype(np.int16)  # Scale to 16-bit range
    
    return hrtf_left, hrtf_right



def apply_hrtf(audio_data, hrtf_left, hrtf_right):
    left_channel = convolve(audio_data, hrtf_left, mode='same')
    right_channel = convolve(audio_data, hrtf_right, mode='same')
    return np.array([left_channel, right_channel]).T

def noise_isolation_and_apply_hrtf(audio_file, hrtf_file_path):
    audio = AudioSegment.from_file(audio_file)
    
    samples = np.array(audio.get_array_of_samples())
    
    if audio.channels == 2:
        samples = samples[::2]  
    reduced_noise = reduce_noise(y=samples, sr=audio.frame_rate)


    hrtf_left, hrtf_right = load_hrtf_filters(hrtf_file_path)


    spatial_audio = apply_hrtf(reduced_noise, hrtf_left, hrtf_right)


    spatial_audio_segment = AudioSegment(
        spatial_audio.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,  # 2 bytes for int16
        channels=2  # Stereo output
    )
    
    output_path = audio_file.rsplit('.', 1)[0] + '_spatial.wav'
    spatial_audio_segment.export(output_path, format='wav')
    return output_path


def translate_text(request):
    translation = None
    audio_url = None
    transcribed_text = None
    detected_lang = None

    if request.method == "POST":
        audio_file = request.FILES.get('audio')
        recorded_audio = request.POST.get('recorded_audio')

        recognizer = sr.Recognizer()
        translator = Translator()

        try:
            hrtf_file_path = r'C:\Users\zozoj\OneDrive\Desktop\NOTES FOR HONORS\COS700\Testing2\classify_speech\cipic-hrtf-database-master\standard_hrir_database\subject_155\hrir_final.mat'

            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    for chunk in audio_file.chunks():
                        temp_file.write(chunk)
                    temp_file.flush()
                    wav_path = convert_to_wav(temp_file.name)
                    spatial_audio_path = noise_isolation_and_apply_hrtf(wav_path, hrtf_file_path)
                    with sr.AudioFile(spatial_audio_path) as source:
                        audio = recognizer.record(source)
            elif recorded_audio:
                audio_data = recorded_audio.split(',')[1]
                audio_bytes = base64.b64decode(audio_data)
                audio_buffer = BytesIO(audio_bytes) 
                
                # Save the audio buffer to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_file.flush()
                    wav_path = convert_to_wav(temp_file.name)
                    spatial_audio_path = noise_isolation_and_apply_hrtf(wav_path, hrtf_file_path)
                    with sr.AudioFile(spatial_audio_path) as source:
                        audio = recognizer.record(source) 
            else:
                raise ValueError("No audio file or recorded audio provided.")

            # Transcribe the audio to text
            transcribed_text = recognizer.recognize_google(audio)

            # Detect the language of the transcribed text
            detected_lang = translator.detect(transcribed_text).lang

            # Translate the transcribed text to the specified target language
            dest_lang = request.POST.get("dest_lang")
            translated = translator.translate(transcribed_text, src=detected_lang, dest=dest_lang)
            translation = translated.text

            # Convert the translated text to speech
            tts = gTTS(text=translation, lang=dest_lang)
            audio_folder = os.path.join(settings.MEDIA_ROOT, 'translation_audio')
            os.makedirs(audio_folder, exist_ok=True)

            audio_output_file = os.path.join(audio_folder, "translation.mp3")
            tts.save(audio_output_file)

            audio_url = f'{settings.MEDIA_URL}translation_audio/translation.mp3'

        except Exception as e:
            transcribed_text = f"Error: {e}"

    return render(request, 'interface.html', {
        'translation': translation,
        'transcribed_text': transcribed_text,
        'detected_lang': detected_lang,  # Include detected language in the response
        'audio_url': audio_url 
    })