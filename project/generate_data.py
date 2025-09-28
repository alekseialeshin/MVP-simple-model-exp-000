import os
import numpy as np
from scipy.io import wavfile

os.makedirs('project/data/real', exist_ok=True)
os.makedirs('project/data/fake', exist_ok=True)

sr = 16000
t = np.linspace(0, 1, sr, False)  # 1 сек

# 5 «реальных»: тон 440 Гц + небольшой шум
for i in range(5):
    tone = 0.5 * np.sin(2*np.pi*440*t)
    noise = 0.05 * np.random.randn(sr)
    x = np.clip(tone + noise, -1, 1)
    wavfile.write(f'project/data/real/real_{i}.wav', sr, (x*32767).astype(np.int16))

# 5 «фейков»: белый шум
for i in range(5):
    x = np.clip(0.5 * np.random.randn(sr), -1, 1)
    wavfile.write(f'project/data/fake/fake_{i}.wav', sr, (x*32767).astype(np.int16))

print("Generated 5 real and 5 fake WAVs.")
