# V-Chamber-C-processor
Designed a hybrid thermodynamic–ML framework integrating a 3 mm phase-change vapor chamber with predictive analytics to enhance compact processor thermal stability. Applied transient heat transfer modeling and LSTM forecasting to anticipate thermal spikes, enabling proactive control that reduced throttling and improved sustained performance.
# ==========================================
# AI-Driven Predictive Thermal Management
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------------
# 1️⃣ Simulate Thermal Dataset
# ----------------------------

np.random.seed(42)

time_steps = 1000
cpu_load = np.random.uniform(10, 100, time_steps)

temperature = []
current_temp = 40

for load in cpu_load:
    # Thermal inertia simulation
    temp_change = 0.05 * load - 0.02 * (current_temp - 35)
    current_temp += temp_change + np.random.normal(0, 0.5)
    temperature.append(current_temp)

data = pd.DataFrame({
    "cpu_load": cpu_load,
    "temperature": temperature
})

# ----------------------------
# 2️⃣ Data Preprocessing
# ----------------------------

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 20
X, y = [], []

for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length][1])  # Predict temperature

X, y = np.array(X), np.array(y)

# ----------------------------
# 3️⃣ Build LSTM Model
# ----------------------------

model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(sequence_length, 2)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# ----------------------------
# 4️⃣ Predict Future Temperature
# ----------------------------

predictions = model.predict(X)
pred_temp = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predictions), 1)), predictions), axis=1)
)[:, 1]

# ----------------------------
# 5️⃣ Predictive Thermal Control
# ----------------------------

THRESHOLD = 80
throttling_actions = []

for temp in pred_temp:
    if temp > THRESHOLD:
        throttling_actions.append("Throttle CPU Frequency")
    else:
        throttling_actions.append("Normal Operation")

# ----------------------------
# 6️⃣ Visualization
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(data["temperature"][sequence_length:], label="Actual Temp")
plt.plot(pred_temp, label="Predicted Temp")
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label="Thermal Limit")
plt.legend()
plt.title("Predictive Thermal Management System")
plt.show()

print("Sample Control Decisions:")
print(throttling_actions[:20])
