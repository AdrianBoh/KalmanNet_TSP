import numpy as np
import matplotlib.pyplot as plt

data = np.load("Results/predictions10.30.25_12-38-07.npz")


observations = data["observations"]      # shape: [N_T, n, T]
predicted_states = data["predicted_states"]  # shape: [N_T, m, T]
ground_truth = data["ground_truth"]       # shape: [N_T, m, T]

idx = 0

# Select first 1000 time steps
T_plot = 1000
pred = predicted_states[idx, :, :T_plot]
gt = ground_truth[idx, :, :T_plot]
obs = observations[idx, :, :T_plot]

# Plot State 0
plt.figure(figsize=(12, 4))
plt.plot(obs[0], label='Observation', color='gray', alpha=0.5)
plt.plot(gt[0], label='Ground Truth', color='green')
plt.plot(pred[0], label='Prediction', color='red', linestyle='--')
plt.title('State 0')
plt.ylabel('Value')
plt.xlabel('Time step')
plt.legend()
plt.tight_layout()

# Plot State 1
plt.figure(figsize=(12, 4))
plt.plot(obs[1], label='Observation', color='gray', alpha=0.5)
plt.plot(gt[1], label='Ground Truth', color='green')
plt.plot(pred[1], label='Prediction', color='red', linestyle='--')
plt.title('State 1')
plt.ylabel('Value')
plt.xlabel('Time step')
plt.legend()
plt.tight_layout()

plt.show()