import librosa as lb
import numpy as np
from sklearn import preprocessing
import soundfile as sf


def convert_to_spectrogram(audio_data):
    """
    Converts the digital data extracted from a WAV file into a magnitude spectrogram
    :param audio_data: The digital data extracted from a WAV file
    :type audio_data: numpy.ndarray
    :return: The magnitude spectrogram of the audio data
    :rtype: numpy.ndarray
    """
    spectogram = lb.stft(audio_data)
    spectogram_mag, _ = lb.magphase(spectogram)
    return spectogram_mag


def convert_to_mel(spectrogram_mag, sr):
    """
    Converts a magnitude spectrogram into a Mel spectrogram
    :param spectogram_mag: Magnitude spectrogram of the audio data
    :type spectogram_mag: numpy.ndarray
    :param sr: The sampling rate of the audio
    :type sr: int
    :return: The Mel spectrogram
    :rtype: numpy.ndarray
    """
    mel_scale_sgram = lb.feature.melspectrogram(S=spectrogram_mag, sr=sr)
    mel_spectogram = lb.amplitude_to_db(mel_scale_sgram, ref=np.min)
    return mel_spectogram

def convert_to_mfcc(audio_data, sr):
    """
    Converts the digital data extracted from a WAV file into Mel-frequency cepstral coefficients (MFCC)
    :param audio_data: The digital data extracted from a WAV file
    :type audio_data: numpy.ndarray
    :param sr: The sampling rate of the audio
    :type sr: int
    :return: The MFCC of the audio data
    :rtype: numpy.ndarray
    """
    mfcc = lb.feature.mfcc(y=audio_data, sr=sr)
    mfcc = preprocessing.scale(mfcc, axis=1)
    return mfcc


def splitAudio_using_VAD(spectogram_mag, audio_data, sr):
    """
    Splits the audio data into speech segments using Voice Activity Detection (VAD)
    :param spectogram_mag: Magnitude spectrogram of the audio data
    :type spectogram_mag: numpy.ndarray
    :param audio_data: The digital data extracted from a WAV file
    :type audio_data: numpy.ndarray
    :param sr: The sampling rate of the audio
    :type sr: int
    :return: None
    """
    f_min = 100
    f_max = 4000
    bins_per_octave = 12

    freq_range = lb.core.cqt_frequencies(
        bins_per_octave=bins_per_octave,
        fmin=f_min,
        n_bins=spectogram_mag.shape[0]
    )
    bin_range = np.where((freq_range >= f_min) & (freq_range <= f_max))[0]

    energy = np.sum(spectogram_mag[bin_range, :], axis=0)

    threshold = np.mean(energy) + np.std(energy)

    speech_segments = np.where(energy > threshold)[0]

    for i, start_idx in enumerate(speech_segments):
        end_idx = speech_segments[i+1] if i < len(speech_segments)-1 else len(audio_data)
        speech_segment = audio_data[start_idx*512:end_idx*512]
        sf.write(f'speech_segment_{i}.wav', speech_segment, sr)