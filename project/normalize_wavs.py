import os, glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOTS = ["project/data/real", "project/data/fake"]
TARGET_SR = 16000

def to_mono(x):
    if x.ndim == 1: return x
    return x.mean(axis=1)

def save_wav(path, sr, x):
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (x*32767).astype(np.int16))

for root in ROOTS:
    for p in glob.glob(os.path.join(root, "*.wav")):
        sr, x = wavfile.read(p)
        x = x.astype(np.float32) / 32767.0
        x = to_mono(x)
        if sr != TARGET_SR:
            g = np.gcd(sr, TARGET_SR)
            up, down = TARGET_SR // g, sr // g
            x = resample_poly(x, up, down)
            sr = TARGET_SR
        save_wav(p, sr, x)
print("Done.")
