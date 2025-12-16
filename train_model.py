import mne
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def bandpower(signal, fs, fmin, fmax):
    freqs, psd = welch(signal, fs, nperseg=fs*2)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.log(np.sum(psd[idx]) + 1e-8)

def extract_features(file_path, label, fs=160):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 40)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, overlap=1.0, preload=True
    )

    X, y = [], []

    for ep in epochs.get_data():
        features = []

        # 🔥 BANDPOWER PER CHANNEL (KEY FIX)
        for ch in ep:
            features.append(bandpower(ch, fs, 4, 8))   # theta
            features.append(bandpower(ch, fs, 8, 13))  # alpha
            features.append(bandpower(ch, fs, 13, 30)) # beta

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)

# -------- SUBJECT-DEPENDENT (S001) --------
X0, y0 = extract_features(
    "../dataset/eeg-motor-movementimagery-dataset-1.0.0/files/S001/S001R01.edf", 0
)
X1, y1 = extract_features(
    "../dataset/eeg-motor-movementimagery-dataset-1.0.0/files/S001/S001R04.edf", 1
)

X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# Balanced split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔥 BALANCED SVM
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred))

joblib.dump(model, "eeg_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model & scaler saved")
