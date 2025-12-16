import mne

file_path = "../dataset/eeg-motor-movementimagery-dataset-1.0.0/files/S001/S001R01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)
print(raw)

raw.plot()
