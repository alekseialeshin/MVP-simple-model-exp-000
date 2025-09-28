import numpy as np
from scipy.io import wavfile
from scipy import signal

def extract_logspec_meanvar(path, nperseg=512, noverlap=256):
    """Лог-спектрограмма -> вектор [mean||var] по времени."""
    sr, data = wavfile.read(path)
    data = data.astype(np.float32) / 32767.0
    _, _, Zxx = signal.stft(data, sr, nperseg=nperseg, noverlap=noverlap)
    log_mag = np.log1p(np.abs(Zxx))
    mean = log_mag.mean(axis=1)
    var  = log_mag.var(axis=1)
    return np.hstack([mean, var])

# --- Опционально: если есть librosa, добавим MFCC/LFCC/CQT ---
try:
    import librosa
    def extract_mfcc(path, sr=16000, n_mfcc=20, hop_length=512):
        y, _ = librosa.load(path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta, delta2]).T

    def extract_logspec_librosa(path, sr=16000, n_fft=1024, hop_length=512):
        y, _ = librosa.load(path, sr=sr, mono=True)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        return np.log1p(np.abs(stft)).T

    def extract_cqt_db(path, sr=16000, n_bins=84):
        y, _ = librosa.load(path, sr=sr, mono=True)
        cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins))
        return librosa.amplitude_to_db(cqt).T

    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False
