import os, glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

SRC = "project/data/real"
DST = "project/data/fake"
os.makedirs(DST, exist_ok=True)

def time_stretch(x, rate=0.85):
    n = int(len(x)*rate)
    return resample(x, n)

for p in glob.glob(os.path.join(SRC, "*.wav")):
    sr, x = wavfile.read(p)
    x = x.astype(np.float32)/32767.0
    y = time_stretch(x, rate=0.85)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x)-len(y)))
    else:
        y = y[:len(x)]
    out = os.path.join(DST, os.path.basename(p).replace(".wav", "_pseudo.wav"))
    wavfile.write(out, sr, (np.clip(y, -1, 1)*32767).astype(np.int16))
print("Pseudo-fakes ready.")
