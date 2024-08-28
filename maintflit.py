import datetime
import os
import re
import threading

import librosa
import matplotlib
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
from kivy.app import App
from kivy.core.audio import SoundLoader
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from matplotlib import pyplot as plt

from audio_recorder import AudioRecorder
from file_logger import FileLogger
from preprocessing import AudioProcessor

matplotlib.use('Agg')

# Define class names
class_names = {
    0: 'chainsaw',
    1: 'dog_bark',
    2: 'engine_idling',
    3: 'gun_shot'
}

os.makedirs('audio', exist_ok=True)
os.makedirs('spectrograms', exist_ok=True)

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
log_file = FileLogger()

# Initialize necessary components
audio_recorder = AudioRecorder()
audio_processor = AudioProcessor()
audio_queue = audio_recorder.get_audio_queue()
stop_event = threading.Event()
recording = False  # Variable to keep track of recording state


class AudioApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.label = Label(text='Press the button to start recording')
        self.start_button = Button(text='Start Recording', on_press=self.start_recording)
        self.stop_button = Button(text='Stop Recording', on_press=self.stop_recording, disabled=True)
        self.log_button = Button(text='View Logs', on_press=self.show_logs)

        self.layout.add_widget(self.label)
        self.layout.add_widget(self.start_button)
        self.layout.add_widget(self.stop_button)
        self.layout.add_widget(self.log_button)

        return self.layout

    def start_recording(self, instance):
        global recording
        self.label.text = 'Recording...'
        self.start_button.disabled = True
        self.stop_button.disabled = False
        stop_event.clear()
        self.recording_thread = threading.Thread(target=audio_recorder.record_audio, args=(stop_event,))
        self.prediction_thread = threading.Thread(target=self.classify_audio)
        self.recording_thread.start()
        self.prediction_thread.start()
        recording = True

    def stop_recording(self, instance):
        global recording
        self.label.text = 'Recording stopped'
        self.start_button.disabled = False
        self.stop_button.disabled = True
        stop_event.set()
        self.recording_thread.join()
        self.prediction_thread.join()
        recording = False

    def classify_audio(self, audio_queue, audio_processor, stop_event):
        while not stop_event.is_set():
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                processed_data = audio_processor.preprocess_audio(audio_data)
                processed_data = np.expand_dims(processed_data, axis=0).astype(np.float32)

                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], processed_data)

                # Run inference
                interpreter.invoke()

                # Get output tensor
                prediction = interpreter.get_tensor(output_details[0]['index'])
                result = np.argmax(prediction, axis=1)[0]
                confidence = round(prediction[0][result] * 100)
                class_detected = class_names[result]
                if confidence > 90:
                    event_description = f"Detected sound: {class_detected} confidence: {confidence}%  " + str(
                        datetime.datetime.now())
                    # Sanitize filenames
                    audio_filename = re.sub(r'[^\w]', '_', event_description) + '.wav'
                    spectrogram_filename = re.sub(r'[^\w]', '_', event_description) + '.png'
                    # Ensure directories exist
                    os.makedirs('audio', exist_ok=True)
                    os.makedirs('spectrograms', exist_ok=True)
                    # Save audio and spectrogram files
                    audio_filepath = os.path.join('audio', audio_filename)
                    spectrogram_filepath = os.path.join('spectrograms', spectrogram_filename)
                    sf.write(audio_filepath, audio_data, samplerate=16000)
                    # Generate and save the spectrogram
                    self.generate_spectrogram(class_detected, audio_data, spectrogram_filepath)
                    log_file.log_event(event_description, audio_filename, spectrogram_filename)

    def generate_spectrogram(self, title, audio_data, output_path, sr=16000):
        # Convert audio data to floating-point format
        audio_data = audio_data.astype(np.float32) / 32768.0
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        plt.figure(figsize=(20, 8))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def show_logs(self, instance):
        log_entries = log_file.get_logs()
        log_layout = GridLayout(cols=1, size_hint_y=None)
        log_layout.bind(minimum_height=log_layout.setter('height'))

        for entry in log_entries:
            label = Label(text=entry.description, size_hint_y=None, height=40)
            log_layout.add_widget(label)
            audio_button = Button(text='Play Audio', size_hint_y=None, height=40)
            audio_button.bind(on_press=lambda x, file=entry.audio_file: self.play_audio(file))
            log_layout.add_widget(audio_button)
            image_button = Button(text='View Spectrogram', size_hint_y=None, height=40)
            image_button.bind(on_press=lambda x, file=entry.spectrogram_file: self.view_spectrogram(file))
            log_layout.add_widget(image_button)

        scrollview = ScrollView(size_hint=(1, None), size=(400, 400))
        scrollview.add_widget(log_layout)

        self.layout.clear_widgets()
        self.layout.add_widget(scrollview)
        back_button = Button(text='Back', on_press=self.back_to_main)
        self.layout.add_widget(back_button)

    def back_to_main(self, instance):
        self.layout.clear_widgets()
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.start_button)
        self.layout.add_widget(self.stop_button)
        self.layout.add_widget(self.log_button)

    def play_audio(self, file):
        sound = SoundLoader.load(os.path.join('audio', file))
        if sound:
            sound.play()

    def view_spectrogram(self, file):
        self.layout.clear_widgets()
        img = Image(source=os.path.join('spectrograms', file))
        self.layout.add_widget(img)
        back_button = Button(text='Back', on_press=self.show_logs)
        self.layout.add_widget(back_button)


if __name__ == '__main__':
    AudioApp().run()
