# /api/emotion.py (Using .h5 SER Model)

import os
import requests
import time
from http.server import BaseHTTPRequestHandler
import json
import soundfile
import librosa
import numpy as np
import tensorflow as tf # Import tensorflow

# --- Load the Model ---
# !!! IMPORTANT: Update this filename to match your .h5 file !!!
model_filename = 'Emotion_Voice_Detection_Model.h5' # <--- UPDATE THIS FILENAME IF NEEDED
# ---

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', model_filename)
model = None # Initialize model as None
emotion_labels = None # Initialize labels as None

try:
    # Use load_model for .h5 files
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Keras model loaded successfully from {model_path}")
        # Define the emotion labels based on the model's training
        # Check the GitHub repo or model details for the correct labels/order if needed
        emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    else:
        print(f"[FATAL ERROR] Model file not found at {model_path}. Make sure '{model_filename}' is in the 'model' folder and the filename is correct.")

except Exception as e:
    print(f"[FATAL ERROR] Error loading Keras model: {e}")
    # Keep model as None if loading fails

class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("[INFO] Function execution started.")
        if model is None or emotion_labels is None:
            print("[ERROR] Model or labels not loaded. Cannot process request.")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_payload = {'error': 'Model could not be loaded on the server. Check Vercel build logs.'}
            self.wfile.write(json.dumps(error_payload).encode())
            return

        temp_audio_path = "/tmp/user_audio.wav" # Define path early for cleanup

        try:
            # 1. Read incoming data from Ultravox
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length))
            # Log the entire payload to help find the real audio URL key
            print(f"[DEBUG] Received full payload: {json.dumps(body, indent=2)}")

            # --- !!! STEP 7 WILL REQUIRE EDITING THIS !!! ---
            # You MUST inspect the Vercel logs from a live test (Step 7)
            # to find the actual key Ultravox uses for the audio URL.
            # Replace 'utterance_audio_url' below with the correct key from your logs.
            audio_url_key = 'utterance_audio_url' # Placeholder key - UPDATE IN STEP 7!
            audio_url = body.get(audio_url_key)
            # --- !!! ------------------------------------- !!! ---

            if not audio_url or not isinstance(audio_url, str) or not audio_url.startswith('http'):
                 print(f"[ERROR] No valid audio URL found in payload field '{audio_url_key}'. Value was: {audio_url}")
                 self.send_response(400)
                 self.send_header('Content-type', 'application/json')
                 self.end_headers()
                 error_payload = {'error': f"No valid audio URL received from Ultravox in expected field '{audio_url_key}'."}
                 self.wfile.write(json.dumps(error_payload).encode())
                 return

            print(f"[INFO] Using audio URL: {audio_url}")

            # 2. Download the audio file temporarily
            print(f"[INFO] Downloading audio from {audio_url}...")
            response = requests.get(audio_url, timeout=15) # Increased download timeout
            response.raise_for_status() # Check for download errors (like 403 Forbidden, 404 Not Found)

            with open(temp_audio_path, 'wb') as f:
                f.write(response.content)
            print(f"[INFO] Audio downloaded successfully to {temp_audio_path}")

            # 3. Extract features from the audio file
            print("[INFO] Extracting features...")
            features = extract_feature(temp_audio_path, mfcc=True, chroma=True, mel=True)
            if features is None:
                raise Exception("Could not extract features from audio file (check format/corruption or file path).")
            # Reshape for Keras model (needs batch dimension)
            features = np.expand_dims(features, axis=0)
            print(f"[INFO] Features extracted successfully. Shape: {features.shape}")

            # 4. Make prediction using the loaded Keras model
            print("[INFO] Making prediction...")
            prediction = model.predict(features)
            predicted_index = np.argmax(prediction[0])
            predicted_emotion = emotion_labels[predicted_index] # Map index to label
            print(f"[INFO] Prediction array: {prediction[0]}")
            print(f"[INFO] Predicted index: {predicted_index}, Emotion: {predicted_emotion}")

            # --- Map to simpler POSITIVE/NEGATIVE/NEUTRAL ---
            sentiment = "NEUTRAL"
            positive_emotions = ["happy", "calm", "surprised"] # Adjust if your model's labels differ
            negative_emotions = ["angry", "sad", "fearful", "disgust"] # Adjust if your model's labels differ

            if predicted_emotion in positive_emotions:
                 sentiment = "POSITIVE"
            elif predicted_emotion in negative_emotions:
                 sentiment = "NEGATIVE"
            # Neutral stays NEUTRAL

            print(f"[INFO] Mapped sentiment: {sentiment}")

            # 5. Send the result back to Ultravox
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response_payload = {'detected_emotion': sentiment.upper()}
            self.wfile.write(json.dumps(response_payload).encode())
            print("[INFO] Successfully sent response to Ultravox.")

        # ... (Error handling remains the same) ...
        except requests.exceptions.Timeout:
            print(f"[ERROR] Timeout downloading audio URL: {audio_url}")
            self.send_response(504); self.send_header('Content-type','application/json'); self.end_headers()
            self.wfile.write(json.dumps({'error': 'Timeout downloading audio file.'}).encode())
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to download/access audio URL {audio_url}: {e}")
            self.send_response(502); self.send_header('Content-type','application/json'); self.end_headers()
            self.wfile.write(json.dumps({'error': f"Failed to download/access audio URL: {e}"}).encode())
        except Exception as e:
            print(f"[FATAL ERROR] An exception occurred during processing: {e}")
            self.send_response(500); self.send_header('Content-type','application/json'); self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        finally:
             # Ensure temporary audio file is cleaned up
            if os.path.exists(temp_audio_path):
                try: os.remove(temp_audio_path); print(f"[INFO] Cleaned up {temp_audio_path}")
                except Exception as e: print(f"[WARNING] Could not remove temp file {temp_audio_path}: {e}")
        return


# --- Feature Extraction Function (Adapted from MITESHPUTHRAN repo) ---
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if X.ndim > 1: X = np.mean(X, axis=1) # Convert to mono
            print(f"[DEBUG] Audio properties: Sample Rate={sample_rate}, Duration={len(X)/sample_rate:.2f}s")

            hop_length = 512
            stft = np.abs(librosa.stft(X, hop_length=hop_length))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, hop_length=hop_length).T, axis=0)
                result = np.hstack((result, chroma_feat))
            if mel:
                mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel_feat))
            print(f"[DEBUG] Feature array shape: {result.shape}")
            return result
    except Exception as e:
        print(f"[ERROR] Librosa/Soundfile failed to process {file_name}: {e}")
        return None