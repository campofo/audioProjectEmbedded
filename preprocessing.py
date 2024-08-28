import numpy as np
import librosa


class AudioProcessor:
    """
    A class used to preprocess audio data for further analysis.

    Methods
    -------
    preprocess_audio(audio_data, rate=16000, target_length=5 * 16000)
        Preprocesses audio data by resampling, padding/truncating to target length, and converting to a log-mel spectrogram.
    """

    @staticmethod
    def preprocess_audio(audio_data, rate=16000, target_length=5 * 16000):
        """
        Preprocesses audio data by resampling, padding/truncating to target length, and converting to a log-mel spectrogram.

        Parameters
        ----------
        audio_data : np.ndarray
            The raw audio data as a NumPy array.
        rate : int, optional
            The sampling rate of the input audio data (default is 16000).
        target_length : int, optional
            The target length of the audio data in samples (default is 80000, which corresponds to 5 seconds at 16000 Hz).

        Returns
        -------
        np.ndarray
            The preprocessed audio data as a log-mel spectrogram, expanded with an additional dimension.
        """
        # Convert audio data to float
        y = audio_data.astype(float)

        # Resample audio if the sampling rate is different from 16000 Hz
        if rate != 16000:
            y = librosa.resample(y, orig_sr=rate, target_sr=16000)

        # Pad or truncate audio to the target length
        if target_length:
            if len(y) < target_length:
                # Pad audio if it's shorter than target length
                padding = target_length - len(y)
                y = np.pad(y, (0, padding), 'constant')
            elif len(y) > target_length:
                # Truncate audio if it's longer than target length
                y = y[:target_length]

        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=16000)

        # Convert mel spectrogram to log scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Expand dimensions to add an additional axis to match the input of the CNN model
        return np.expand_dims(log_mel_spectrogram, axis=-1)
