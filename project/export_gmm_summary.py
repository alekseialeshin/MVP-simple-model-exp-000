import os, json
from joblib import load

MODEL_DIR = "project/models"
OUT_PATH = "project/models/gmm_summary.json"

def summarize(path):
    gmm = load(path)
    return {
        "path": path,
        "n_components": int(gmm.n_components),
        "covariance_type": gmm.covariance_type,
        "n_features": int(gmm.means_.shape[1]),
        "weights": gmm.weights_.round(6).tolist(),
        "means_head": gmm.means_.round(6)[:2].tolist(),
    }

real = summarize(os.path.join(MODEL_DIR, "gmm_real.joblib"))
fake = summarize(os.path.join(MODEL_DIR, "gmm_fake.joblib"))
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"real": real, "fake": fake}, f, ensure_ascii=False, indent=2)
print(f"Saved: {OUT_PATH}")
