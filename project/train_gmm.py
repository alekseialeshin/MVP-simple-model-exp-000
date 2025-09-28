import os, glob
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve
from joblib import dump
from features import extract_logspec_meanvar  # базовый вариант без librosa

REAL_DIR = "project/data/real"
FAKE_DIR = "project/data/fake"
MODEL_DIR = "project/models"

def load_set(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    X = [extract_logspec_meanvar(p) for p in files]
    return np.array(X), files

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fpr - fnr))
    return max(fpr[i], fnr[i])

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_real, _ = load_set(REAL_DIR)
    X_fake, _ = load_set(FAKE_DIR)
    if len(X_real)==0 or len(X_fake)==0:
        raise RuntimeError("Нет данных в project/data/real или project/data/fake")

    gmm_real = GaussianMixture(n_components=2, covariance_type="diag", random_state=42).fit(X_real)
    gmm_fake = GaussianMixture(n_components=2, covariance_type="diag", random_state=42).fit(X_fake)

    s_real = gmm_real.score_samples(X_real) - gmm_fake.score_samples(X_real)
    s_fake = gmm_real.score_samples(X_fake) - gmm_fake.score_samples(X_fake)
    scores = np.concatenate([s_real, s_fake])
    labels = np.concatenate([np.ones(len(s_real)), np.zeros(len(s_fake))])

    eer = compute_eer(scores, labels)
    print(f"Примеры (real): {len(s_real)} Примеры (fake): {len(s_fake)}")
    print("Пример LLR(real):", np.round(s_real[:3], 2))
    print("Пример LLR(fake):", np.round(s_fake[:3], 2))
    print(f"EER = {eer:.4f}")

    dump(gmm_real, os.path.join(MODEL_DIR, "gmm_real.joblib"))
    dump(gmm_fake, os.path.join(MODEL_DIR, "gmm_fake.joblib"))
    print("Модели сохранены в project/models/gmm_real.joblib и gmm_fake.joblib")

if __name__ == "__main__":
    main()
