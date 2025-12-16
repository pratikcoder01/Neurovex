import mne

file_path = "../dataset/eeg-motor-movementimagery-dataset-1.0.0/files/S001/S001R01.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

# Band-pass filter
raw.filter(0.5, 40)

# Create epochs WITH preload
epochs = mne.make_fixed_length_epochs(
    raw,
    duration=2.0,
    overlap=1.0,
    preload=True
)

print("Number of epochs created:", len(epochs))

epochs.plot()
