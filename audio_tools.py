from librosa import load
from librosa.feature import melspectrogram

#ToDo: cleanup
def mel_spectrogram(wav_file):
    # Read out audio range and sample rate of wav file
    audio_range, sample_rate = load(path=wav_file, sr=None)
    nperseg = int(10 * sample_rate / 1000)

    # NOTE: nperseg MUST be an int before handing it over to liberosa's function
    mel_spectrogram = melspectrogram(y=audio_range, sr=sample_rate, n_fft=1024, hop_length=nperseg)

    # Compress the mel spectrogram to the human dynamic range
    for i in range(mel_spectrogram.shape[0]):
        for j in range(mel_spectrogram.shape[1]):
            mel_spectrogram[i, j] = dyn_range_compression(mel_spectrogram[i, j])

    return mel_spectrogram