import os
import time
import queue
import threading
import numpy as np
import torchaudio
import torch
import speech_recognition as sr
import whisperx
from datetime import datetime
from transformers import MarianMTModel, MarianTokenizer, pipeline, AutoModelForSequenceClassification, AutoTokenizer
from TTS.api import TTS
import librosa
import parselmouth
from parselmouth.praat import call
import glob
import sounddevice as sd
from scipy.io.wavfile import read
import argparse
import re
from io import BytesIO
from scipy.io.wavfile import write
import soundfile as sf
import tempfile


SAMPLE_RATE = 16000
RECORD_TIMEOUT = 5
PHRASE_TIMEOUT = 2
CHUNK_OVERLAP = 0.3
XTTS_SAMPLE_RATE = 24000

GERMAN_LABELS = {
    0: "anger",
    1: "fear",
    2: "disgust",
    3: "sadness",
    4: "joy",
    5: "neutral"
}
ITALIAN_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}



# Queue
data_queue = queue.Queue()
urgency_audio_queue = queue.Queue()
urgency_results_queue = queue.Queue()
emotion_queue = queue.Queue()
emotion_results_queue = queue.Queue()
transcription_queue = queue.Queue()
translation_queue = queue.Queue()
tts_queue = queue.Queue()
playback_queue = queue.Queue()

tts_counter = 0
tts_lock = threading.Lock()

#output_folder = "output_tts"
#for file in glob.glob(os.path.join(output_folder, "*")):
#    os.remove(file)


def calculate_phrase_lengths(audio_file):
    #compute phrase lenght (a phrase are word between silence pause)
    y, sr = librosa.load(audio_file, sr=16000)
    non_silent_intervals = librosa.effects.split(y, top_db=30)

    phrase_lengths = [
        (non_silent_intervals[i][0] - non_silent_intervals[i - 1][1]) / sr
        for i in range(1, len(non_silent_intervals))
    ]
    return phrase_lengths

def extract_urgency_features(file_path, sr=16000):
    y_librosa, sr = librosa.load(file_path, sr=sr)
    duration = librosa.get_duration(y=y_librosa, sr=sr)
    non_silent_intervals = librosa.effects.split(y_librosa, top_db=30)
    speaking_time = sum([(end - start) / sr for start, end in non_silent_intervals])

    snd = parselmouth.Sound(file_path)

    pitch = snd.to_pitch()
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    pause_durations = [
        (non_silent_intervals[i][0] - non_silent_intervals[i - 1][1]) / sr
        for i in range(1, len(non_silent_intervals))
    ]
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
    num_pauses = len(pause_durations)
    avg_phrase_length = np.mean(pause_durations) if pause_durations else 0
    rms_amplitude = librosa.feature.rms(y=y_librosa).mean()
    pitch_range = f0_max - f0_min

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "pitch_range": pitch_range,
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
    rms_amplitude_threshold = 0.04

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
def audio_capture_worker(record_timeout, phrase_timeout, urgency_from):
    #audio_ready_event.wait()
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

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Audio capture begin.")
    previous_audio = None
    while True:
        now = datetime.utcnow()
        if not data_queue.empty():
            #phrase_complete = False
            #if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                #phrase_complete = True
            #phrase_time = now

            audio_data = b''.join(data_queue.queue)


            if previous_audio:
                audio_data = previous_audio + audio_data

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            # start_time_capture = time.time()
            transcription_queue.put(audio_np)
            if urgency_from in ("audio", "both"):
                urgency_audio_queue.put(audio_np)

            # end_time_capture = time.time()
            # print(f"[DEBUG] Acquisizione audio completata in {end_time_capture - start_time_capture:.2f} secondi.")

            # Overlap
            previous_audio = audio_data[-int(SAMPLE_RATE * CHUNK_OVERLAP):]
            data_queue.queue.clear()
        else:
            time.sleep(0.1)


def urgency_worker():
    while True:
        audio_np = urgency_audio_queue.get()
        try:
            print("[DEBUG] Start prosody analysis...")
            start_urgency = time.time()

            buffer = BytesIO()
            sf.write(buffer, audio_np.T, SAMPLE_RATE, format='WAV')
            buffer.seek(0)

            if len(audio_np.shape) == 1:
                audio_np = np.expand_dims(audio_np, axis=1)
            audio_np = audio_np.astype(np.float32)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
                sf.write(tmp_file, audio_np, SAMPLE_RATE, format='WAV')

            features = extract_urgency_features(temp_path)
            classification = classify_urgency(features)
            end_urgency = time.time()

            urgency_results_queue.put(classification)
            print(f"Audio classified as: {classification} in {end_urgency - start_urgency:.2f} seconds.")
            #print(f"Features extracted: {features}") #deccoment this line to see the specifics features

            os.remove(temp_path)

        except Exception as e:
            print(f"[ERRORE] Classificazione urgenza fallita: {e}")
        finally:
            urgency_audio_queue.task_done()



# Worker for STT
def stt_worker(whisper_model, src_lan):
    while True:
        audio_np = transcription_queue.get()
        start_time_stt = time.time()
        try:
            print("Start transcription...")
            result = whisper_model.transcribe(audio_np, batch_size=16, language=src_lan)
            segments = result.get("segments", [])
            if not segments:
                print("[WARNING] No segmetns found: empty audio, skipping chunk.")
                continue
            transcription_text = " ".join([seg["text"] for seg in result["segments"]])
            print("Text transcribed:", transcription_text)
            translation_queue.put(transcription_text)
            emotion_queue.put(transcription_text)
        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            transcription_queue.task_done()
            end_time_stt = time.time()
            print(f"Trasncription completed in {end_time_stt - start_time_stt:.2f} seconds.")

def emotion_recognition_worker(emotion_models, src_lan):
    while True:
        transcription = emotion_queue.get()
        start_time = time.time()
        if src_lan == "en" and "classifier_en" in emotion_models:
            result = emotion_models["classifier_en"](transcription)
            emotion = result[0]["label"] if result else "neutral"

        elif src_lan == "fr" and "model_fr" in emotion_models and "tokenizer_fr" in emotion_models:
            inputs = emotion_models["tokenizer_fr"](transcription, return_tensors="pt", padding=True, truncation=True,
                                                    max_length=512).to("cuda")
            outputs = emotion_models["model_fr"](**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_index = prediction.argmax().item()
            labels = emotion_models["model_fr"].config.id2label
            emotion = labels[predicted_index] if predicted_index in labels else str(predicted_index)

        elif src_lan == "de" and "model_de" in emotion_models and "tokenizer_de" in emotion_models:
            inputs = emotion_models["tokenizer_de"](transcription, return_tensors="pt", padding=True, truncation=True,
                                                    max_length=512).to("cuda")
            outputs = emotion_models["model_de"](**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_index = prediction.argmax().item()
            emotion = GERMAN_LABELS.get(predicted_index, f"Unknown (LABEL_{predicted_index})")
        elif src_lan == "it" and "model_it" in emotion_models and "tokenizer_it" in emotion_models:
            inputs = emotion_models["tokenizer_it"](transcription, return_tensors="pt", padding=True, truncation=True,
                                                    max_length=512).to("cuda")
            outputs = emotion_models["model_it"](**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_index = prediction.argmax().item()
            emotion = ITALIAN_LABELS.get(predicted_index, f"Unknown (LABEL_{predicted_index})")
        else:
            emotion = "unknown"

        emotion_time = time.time() - start_time
        print(f"The chunk have been recognised with the emotion of:{emotion} in {emotion_time}")
        emotion_results_queue.put(emotion)
        emotion_queue.task_done()
        #return {"emotion": emotion, "emotion_time": emotion_time}


# Worker per la traduzione
def translation_worker(translation_model, translation_tokenizer, trg_lan, device):
    while True:
        transcription = translation_queue.get()
        start_time_translation = time.time()
        try:
            transcription = f"{transcription.strip()}"
            modified_text = re.sub(r'[,.!]', '', transcription)
            print(f"Start transaltion: {modified_text}")
            src_text = f">>{trg_lan}<< {modified_text}"
            translated = translation_model.generate(
                **translation_tokenizer(src_text, return_tensors="pt", padding=True).to(device)
            )
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
            if translated_text:
                translated_text = translated_text.replace('.', ',')
            if translated_text.strip() == "==References====External links==":
                print("Translation produced a default placeholder. Using original transcription as fallback.")
                translated_text = translated_text.strip()
            print(f"Translation completed: {transcription}")
            tts_queue.put(translated_text)
        except Exception as e:
            print(f"Translation error: {e}")
        finally:
            translation_queue.task_done()
            end_time_translation = time.time()
            print(f"Translation completed in {end_time_translation - start_time_translation:.2f} secondi.")

# Worker per TTS
def tts_worker(tts_model, trg_lan, urgency_from):
    global tts_counter
    while True:
        text = tts_queue.get()
        try:
            urgency_label = urgency_results_queue.get_nowait()
        except queue.Empty:
            urgency_label = None

        try:
            emotion_label = emotion_results_queue.get_nowait()
        except queue.Empty:
            emotion_label = None

        is_urgent = False
        if urgency_from == "audio":
            is_urgent = (urgency_label == "URGENT")
        elif urgency_from == "text":
            is_urgent = (emotion_label in ("anger", "fear"))
        elif urgency_from == "both":
            is_urgent = (urgency_label == "URGENT") or (emotion_label in ("anger", "fear"))

        start_time_tts = time.time()
        try:
            print(f"TTS for the text: {text}")
            with tts_lock:
                file_index = tts_counter
                tts_counter += 1
            output_path = f"output_tts/output_tts_{file_index}.wav"
            if is_urgent:
                multi_speaker = ["speakers/my_urgent_audio_1.wav", "speakers/my_urgent_audio_2.wav"]
            else:
                multi_speaker = ["speakers/my_not_urgent_audio_1.wav", "speakers/my_not_urgent_audio_2.wav"]

            wav = tts_model.tts(text=text, speaker_wav=multi_speaker, language=trg_lan)
            wav = np.asarray(wav, dtype=np.float32)
            buffer = BytesIO()
            write(buffer, XTTS_SAMPLE_RATE, wav)
            buffer.seek(0)
            print(f"TTS output saved as: {output_path}")
            playback_queue.put((buffer, XTTS_SAMPLE_RATE))
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            tts_queue.task_done()
            end_time_tts = time.time()
            print(f"TTS completed in {end_time_tts - start_time_tts:.2f} seconds.")


def audio_playback_worker():
    while True:
        item = playback_queue.get()
        try:
            if isinstance(item, tuple):
                buffer, samplerate = item
                buffer.seek(0)
                _, data = read(buffer)
                print("Playing buffered audio...")
                sd.play(data, samplerate=samplerate)
                sd.wait()
            else:
                print("Invalid playback item format.")
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            playback_queue.task_done()

def start_pipeline(args):
    src_lan = args.src
    trg_lan = args.trg
    chunk_duration = args.chunk_duration
    urgency_from = args.urgency_from
    emotion_models = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32"  # change to 32 for server test
    whisper_model = whisperx.load_model(
        "base",
        device=device,
        compute_type=compute_type,
        language=src_lan,
        asr_options={
            "max_new_tokens": 500,
            "clip_timestamps": True,
            "hallucination_silence_threshold": 0.5,
            "multilingual": True,
            "hotwords": None,
        }
    )
    print("Warming up WhisperX with silent audio...")
    silence = np.zeros(int(SAMPLE_RATE), dtype=np.float32)
    _ = whisper_model.transcribe(silence, batch_size=1, language=src_lan)
    print("WhisperX warm-up complete.")

    print("Loading emotional classification model")
    if src_lan == "en":
        emotion_models["classifier_en"] = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier",
                                                   device=device)
    elif src_lan == "fr":
        fr_model_name = "astrosbd/french_emotion_camembert"
        emotion_models["tokenizer_fr"] = AutoTokenizer.from_pretrained(fr_model_name)
        emotion_models["model_fr"] = AutoModelForSequenceClassification.from_pretrained(fr_model_name, from_tf=True).to(
            device)
    elif src_lan == "de":
        de_model_name = "visegradmedia-emotion/Emotion_RoBERTa_german6_v7"
        emotion_models["tokenizer_de"] = AutoTokenizer.from_pretrained(de_model_name)
        emotion_models["model_de"] = AutoModelForSequenceClassification.from_pretrained(de_model_name).to(device)
    elif src_lan == "it":
        it_model_name = "aiknowyou/it-emotion-analyzer"
        emotion_models["tokenizer_it"] = AutoTokenizer.from_pretrained(it_model_name)
        emotion_models["model_it"] = AutoModelForSequenceClassification.from_pretrained(it_model_name).to(device)
    print(f"Emotional recognition model loaded for language: {src_lan}")

    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    translation_model_name = f"Helsinki-NLP/opus-mt-{src_lan}-{trg_lan}"
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

    print("Loading model complete")

    threading.Thread(target=audio_capture_worker, args=(RECORD_TIMEOUT, PHRASE_TIMEOUT, urgency_from,), daemon=True).start()
    threading.Thread(target=stt_worker, daemon=True, args=(whisper_model, src_lan, )).start()
    threading.Thread(target=translation_worker, daemon=True, args=(translation_model, translation_tokenizer, trg_lan, device,)).start()
    threading.Thread(target=tts_worker, daemon=True, args=(tts_model, trg_lan, urgency_from, )).start()
    threading.Thread(target=audio_playback_worker, daemon=True).start()

    if urgency_from in ("audio", "both"):
        threading.Thread(target=urgency_worker, daemon=True).start()
        print(f"Urgency worker avviato (urgency_from={urgency_from})")
    else:
        print(f"Urgency worker disabilitato (urgency_from={urgency_from})")
    if urgency_from in ("text", "both"):
        threading.Thread(target=emotion_recognition_worker, daemon=True, args=(emotion_models, src_lan,)).start()
        print(f"Emotion recognition worker avviato (urgency_from={urgency_from})")
    else:
        print(f"Emotion recognition worker disabilitato (urgency_from={urgency_from})")

    print("System started. Press Ctrl+C to terminate.")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="en", help="Source Language (es. 'en, fr, de, it')")
    parser.add_argument("--trg", type=str, default="it", help="Destination Language (es. 'en, fr, de')")
    parser.add_argument("--chunk_duration", type=int, default=5, help="Time for each chunk (suggested: 5 seconds)")
    parser.add_argument("--urgency_from", type=str, default="both", choices=["audio", "text", "both"],
                        help="Source from which the urgency recognition will be performed")
    args = parser.parse_args()


    start_pipeline(args)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("System terminated.")

if __name__ == "__main__":
    main()
