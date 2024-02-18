import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define functions to generate different signal types
def generate_sine_wave(f0, fs, duration):
    t = np.linspace(0, duration, int(fs * duration))
    return np.sin(2 * np.pi * f0 * t)

def generate_square_wave(f0, fs, duration):
    t = np.linspace(0, duration, int(fs * duration))
    return np.sign(np.sin(2 * np.pi * f0 * t))

def generate_noise(fs, duration):
    return np.random.randn(int(fs * duration))

# Create data
fs = 1000  # Sampling frequency
duration = 1  # Signal duration in seconds
n_samples = 100  # Number of samples per signal type

signal_types = ["sine", "square", "noise"]
X = []
y = []
for i in range(10000):
    for signal_type in signal_types:
        if signal_type == "sine":
            data = generate_sine_wave(50, fs, duration)
        elif signal_type == "square":
            data = generate_square_wave(50, fs, duration)
        else:
            data = generate_noise(fs, duration)
        X.append(data)
        y.append(signal_type)

X = np.array(X).reshape(-1, len(X[0]))
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LogisticRegression model
model = LogisticRegression(multi_class="ovr", solver="lbfgs")
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


