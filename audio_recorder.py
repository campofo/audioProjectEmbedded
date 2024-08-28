import pyaudio
from queue import Queue
import numpy as np


class AudioRecorder:
    """
    A class used to record audio using PyAudio and store it in a queue.

    Attributes
    ----------
    FORMAT : int
        The audio format (default is 16-bit PCM).
    CHANNELS : int
        The number of audio channels (default is mono).
    RATE : int
        The sampling rate (default is 44100 samples per second).
    CHUNK : int
        The number of frames per buffer (default is 1024).
    RECORD_SECONDS : int
        The length of each recording in seconds (default is 5 seconds).
    WAVE_OUTPUT_FILENAME : str
        The output filename for the recording (default is "file.wav").
    audio_queue : Queue
        A queue to store recorded audio data.
    audio : PyAudio
        An instance of PyAudio.
    stream : Stream
        The audio stream for recording.

    Methods
    -------
    record_audio(stop_event)
        Records audio until the stop_event is set.
    get_audio_queue()
        Returns the queue containing recorded audio data.
    """

    def __init__(self):
        """
        Initializes the AudioRecorder with default parameters.
        """
        self.FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.CHANNELS = 1  # Number of audio channels (mono)
        self.RATE = 44100  # Sampling rate (samples per second)
        self.CHUNK = 1024  # Number of frames per buffer
        self.RECORD_SECONDS = 5  # Length of each recording in seconds
        self.WAVE_OUTPUT_FILENAME = "file.wav"  # Output filename for the recording (if needed)

        # Queue to store recorded audio data
        self.audio_queue = Queue()

        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        self.stream = None  # Audio stream

    def record_audio(self, stop_event):
        """
        Records audio and stores it in the audio queue.

        Parameters
        ----------
        stop_event : threading.Event
            An event that, when set, stops the recording.
        """
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
        print("Recording...")
        while not stop_event.is_set():
            frames = []
            try:
                for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.int16))
                audio_data = np.hstack(frames)
                self.audio_queue.put(audio_data)
            except IOError as e:
                print(f"Error recording audio: {e}")
                continue
        print("Recording stopped.")

        self.stream.stop_stream()
        self.stream.close()

    def get_audio_queue(self):
        """
        Returns the queue containing recorded audio data.

        Returns
        -------
        Queue
            A queue with recorded audio data.
        """
        return self.audio_queue
