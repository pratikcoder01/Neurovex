import mne
import numpy as np
from scipy.signal import welch

# Load and preprocess EEG
file_path = "../dataset/eeg-motor-movementimagery-dataset-1.0.0/files/S001/S001R01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)
raw.filter(0.5, 40)

epochs = mne.make_fixed_length_epochs(
    raw,
    duration=2.0,
    overlap=1.0,
    preload=True
)

def extract_features(epoch, fs=160):
    signal = epoch.mean(axis=0)

    mean = np.mean(signal)
    variance = np.var(signal)

    freqs, psd = welch(signal, fs)

    alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
    beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])

    return [mean, variance, alpha_power, beta_power]

# Extract features from all epochs
X = []
for epoch in epochs.get_data():
    X.append(extract_features(epoch))

X = np.array(X)

print("Feature matrix shape:", X.shape)
print("First feature vector:", X[0])
