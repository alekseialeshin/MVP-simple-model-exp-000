import sys, os
import numpy as np
from joblib import load
from features import extract_logspec_meanvar

MODEL_DIR = "project/models"

def main():
    if len(sys.argv) < 2:
        print("Использование: python project/predict.py path/to/file.wav")
        sys.exit(1)
    wav_path = sys.argv[1]
    if not os.path.isfile(wav_path):
        print(f"Файл не найден: {wav_path}")
        sys.exit(1)

    gmm_real = load(os.path.join(MODEL_DIR, "gmm_real.joblib"))
    gmm_fake = load(os.path.join(MODEL_DIR, "gmm_fake.joblib"))

    x = extract_logspec_meanvar(wav_path).reshape(1, -1)
    llr = (gmm_real.score_samples(x) - gmm_fake.score_samples(x))[0]  # берём скаляр

    verdict = "real" if llr > 0 else "fake"
    print(f"LLR={llr:.2f} → verdict: {verdict}")

if __name__ == "__main__":
    main()
