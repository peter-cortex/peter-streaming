import os
import time
import queue
import threading
import numpy as np
import torchaudio
import whisper
import torch
import speech_recognition as sr
import whisperx
from datetime import datetime, timedelta
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS  #Coqui TTS
import io
import librosa
import parselmouth
from parselmouth.praat import call
import glob


SAMPLE_RATE = 16000
RECORD_TIMEOUT = 5
PHRASE_TIMEOUT = 2
CHUNK_OVERLAP = 0.3
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "it"

audio_ready_event = threading.Event()

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" #change to 32 for server test
whisper_model = whisperx.load_model(
    "base",
    device=device,
    compute_type=compute_type,
    language=SOURCE_LANGUAGE,
    asr_options={
        "max_new_tokens": 500,
        "clip_timestamps": True,
        "hallucination_silence_threshold": 0.5,
        "multilingual": True,
        "hotwords": None,
    }
)

tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


translation_model_name = f"Helsinki-NLP/opus-mt-{SOURCE_LANGUAGE}-{TARGET_LANGUAGE}"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

print("Loading model complete")
audio_ready_event.set()

# Queue
data_queue = queue.Queue()
urgency_queue = queue.Queue()
transcription_queue = queue.Queue()
translation_queue = queue.Queue()
tts_queue = queue.Queue()

tts_counter = 0
tts_lock = threading.Lock()

output_folder = "output_tts"
for file in glob.glob(os.path.join(output_folder, "*")):
    os.remove(file)


def calculate_phrase_lengths(audio_file):
    #compute phrase lenght (a phrase are word between silence pause)
    y, sr = librosa.load(audio_file, sr=16000)
    non_silent_intervals = librosa.effects.split(y, top_db=30)

    phrase_lengths = [
        (non_silent_intervals[i][0] - non_silent_intervals[i - 1][1]) / sr
        for i in range(1, len(non_silent_intervals))
    ]
    return phrase_lengths

def extract_urgency_features(audio_file):
    #extraction of prosody features
    y, sr = librosa.load(audio_file, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    speaking_time = sum([(end - start) / sr for start, end in non_silent_intervals])

    snd = parselmouth.Sound(audio_file)
    pitch = snd.to_pitch()
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    #speech_rate = num_words / (duration / 60) if duration > 0 else 0
    #articulation_rate = num_words / (speaking_time / 60) if speaking_time > 0 else 0

    pause_durations = [
        (non_silent_intervals[i][0] - non_silent_intervals[i - 1][1]) / sr
        for i in range(1, len(non_silent_intervals))
    ]
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
    num_pauses = len(pause_durations)

    phrase_lengths = calculate_phrase_lengths(audio_file)
    avg_phrase_length = np.mean(phrase_lengths) if phrase_lengths else 0

    pitch_range = f0_max - f0_min
    rms_amplitude = librosa.feature.rms(y=y).mean()

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "pitch_range": pitch_range,
        #"speech_rate": speech_rate,
        #"articulation_rate": articulation_rate,
        "avg_pause_duration": avg_pause_duration,
        "num_pauses": num_pauses,
        "avg_phrase_length": avg_phrase_length,
        "rms_amplitude": rms_amplitude,
        "speaking_time": speaking_time,
        "duration": duration,
    }

def classify_urgency(features):
    """
    Classifica l'urgenza di un audio basandosi sulle feature estratte.
    """
    f0_mean_threshold = 270
    f0_std_threshold = 50
    pitch_range_threshold = 50
    #speech_rate_threshold = 100
    #articulation_rate_threshold = 120
    avg_pause_duration_threshold = 0.4
    avg_phrase_length_threshold = 2.0
    rms_amplitude_threshold = 0.06

    is_urgent = (
        features["f0_mean"] > f0_mean_threshold and
        features["f0_std"] > f0_std_threshold and
        features["pitch_range"] > pitch_range_threshold and
        #features["speech_rate"] > speech_rate_threshold and
        #features["articulation_rate"] > articulation_rate_threshold and
        features["avg_pause_duration"] < avg_pause_duration_threshold and
        features["avg_phrase_length"] < avg_phrase_length_threshold and
        features["rms_amplitude"] > rms_amplitude_threshold
    )

    return "URGENT" if is_urgent else "NOT URGENT"




# Audio recording and chunking
def audio_capture_worker(record_timeout, phrase_timeout):
    audio_ready_event.wait()
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    phrase_time = None
    source = sr.Microphone(sample_rate=SAMPLE_RATE)
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout + CHUNK_OVERLAP)
    print("Audio capture begin.")
    previous_audio = None
    while True:
        now = datetime.utcnow()
        if not data_queue.empty():
            phrase_complete = False
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = True
            phrase_time = now

            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            if previous_audio:
                audio_data = previous_audio + audio_data

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            # start_time_capture = time.time()
            transcription_queue.put(audio_np)
            urgency_queue.put(audio_np)
            # end_time_capture = time.time()
            # print(f"[DEBUG] Acquisizione audio completata in {end_time_capture - start_time_capture:.2f} secondi.")

            # Overlap
            previous_audio = audio_data[-int(SAMPLE_RATE * CHUNK_OVERLAP):]
        else:
            time.sleep(0.1)


def urgency_worker():
    while True:
        audio_np = urgency_queue.get()
        try:
            print("[DEBUG] Start prosody analysis...")
            start_urgency = time.time()
            temp_filename = f"temp_audio_{int(time.time())}.wav"

            if len(audio_np.shape) == 1:
                audio_np = np.expand_dims(audio_np, axis=0)

            audio_tensor = torch.tensor(audio_np, dtype=torch.float32)

            torchaudio.save(temp_filename, audio_tensor , SAMPLE_RATE)

            # Estrarre le feature
            features = extract_urgency_features(temp_filename)
            classification = classify_urgency(features)
            end_urgency = time.time()

            print(f"Audio classified as: {classification} in {end_urgency - start_urgency:.2f} seconds.")
            #print(f"Features extracted: {features}") #deccoment this line to see the specifics features

            os.remove(temp_filename)

        except Exception as e:
            print(f"[ERRORE] Classificazione urgenza fallita: {e}")
        finally:
            urgency_queue.task_done()



# Worker for STT
def stt_worker():
    while True:
        audio_np = transcription_queue.get()
        start_time_stt = time.time()
        try:
            print("Start transcription...")
            result = whisper_model.transcribe(audio_np, batch_size=16, language=SOURCE_LANGUAGE)
            transcription_text = " ".join([seg["text"] for seg in result["segments"]])
            print("Text transcribed:", transcription_text)
            translation_queue.put(transcription_text)
        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            transcription_queue.task_done()
            end_time_stt = time.time()
            print(f"Trasncription completed in {end_time_stt - start_time_stt:.2f} seconds.")

# Worker per la traduzione
def translation_worker():
    while True:
        text = translation_queue.get()
        start_time_translation = time.time()
        try:
            print(f"Start transaltion: {text}")
            src_text = f">>{TARGET_LANGUAGE}<< {text}"
            translated = translation_model.generate(
                **translation_tokenizer(src_text, return_tensors="pt", padding=True).to(device)
            )
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
            print(f"Translation completed: {translated_text}")
            tts_queue.put(translated_text)
        except Exception as e:
            print(f"Translation error: {e}")
        finally:
            translation_queue.task_done()
            end_time_translation = time.time()
            print(f"Translation completed in {end_time_translation - start_time_translation:.2f} secondi.")

# Worker per TTS: each audio chunk can be found in the output_tts folder
def tts_worker():
    global tts_counter
    while True:
        text = tts_queue.get()
        start_time_tts = time.time()
        try:
            print(f"TTS for the text: {text}")
            with tts_lock:
                file_index = tts_counter
                tts_counter += 1
            output_path = f"output_tts/output_tts_{file_index}.wav"
            speaker_tts = "speaker/it_w.mp3"
            tts_model.tts_to_file(text=text, language=TARGET_LANGUAGE ,file_path=output_path, speaker_wav=speaker_tts)
            print(f"TTS output saved as: {output_path}")
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            tts_queue.task_done()
            end_time_tts = time.time()
            print(f"TTS completed in {end_time_tts - start_time_tts:.2f} seconds.")


def main():
    threading.Thread(target=audio_capture_worker, args=(RECORD_TIMEOUT, PHRASE_TIMEOUT), daemon=True).start()
    threading.Thread(target=stt_worker, daemon=True).start()
    threading.Thread(target=urgency_worker, daemon=True).start()
    threading.Thread(target=translation_worker, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()

    print("System started. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("System terminated.")

if __name__ == "__main__":
    main()
