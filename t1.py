import numpy as np

# Function to calculate velocity from two measurements
def calculate_velocity(measurement1, measurement2, delta_t):
    velocity = (measurement2 - measurement1) / delta_t
    return velocity

# Function to perform association using JPDA
def jpda_association(x_pred, P_pred, measurements, H, R, G, W, validation_gate):
    num_targets = len(x_pred)
    num_measurements = len(measurements)
    
    association_probabilities = np.zeros((num_targets + 1, num_measurements))  # +1 for clutter
    clutter_likelihood = 1 / (np.sqrt((2 * np.pi) ** H.shape[0] * np.linalg.det(R)))
    
    for i in range(num_targets):
        y_pred = np.dot(H, x_pred[i])
        S = np.dot(H, np.dot(P_pred[i], H.T)) + R
        inv_S = np.linalg.inv(S)
        mahalanobis_dist = np.sum(((measurements - y_pred) @ inv_S) * (measurements - y_pred), axis=1)
        association_probabilities[i] = np.exp(-0.5 * mahalanobis_dist) / np.sqrt(np.linalg.det(2 * np.pi * S))
    
    clutter_probabilities = clutter_likelihood * np.ones(num_measurements)
    
    association_probabilities[-1] = clutter_probabilities
    
    associations = []
    for j in range(num_measurements):
        if np.any(association_probabilities[:, j] > validation_gate):
            max_prob_index = np.argmax(association_probabilities[:, j])
            associations.append((max_prob_index, j))
    
    return associations

# Sample Inputs
F = np.eye(6)  # State transition matrix (assuming no acceleration)
H = np.eye(6)  # Measurement mapping matrix
G = np.eye(6)  # Process noise scaling matrix
W = np.eye(6) * 0.1  # Process noise covariance matrix
Q = np.eye(6) * 0.01  # Process noise covariance
R = np.eye(6) * 0.1  # Measurement noise covariance

x_init = np.zeros(6)  # Initial state (assuming no prior information)
P_init = np.eye(6) * 0.1  # Initial covariance
validation_gate = 0.01  # Validation gate threshold

# Sample measurements (range, azimuth, elevation, time)
measurements = np.array([
    [10, 45, 30, 1],  # M1
    [15, 50, 35, 2],  # M2
    [20, 55, 40, 3]   # M3
])

# Step 1: Initialize with the first measurement M1
x_pred = measurements[0]

# Step 2: Initialize with the second measurement M2
x_pred_prev = measurements[1]

# Step 3: Calculate velocity from step 1 and 2
velocity = calculate_velocity(x_pred_prev, x_pred, measurements[1, -1] - measurements[0, -1])

# Step 4: Get measurement M3 and delta t
delta_t = measurements[2, -1] - measurements[1, -1]
measurement_with_velocity = np.concatenate((measurements[1], velocity))

# Step 5: Perform association using JPDA
associations = jpda_association(np.array([x_pred]), np.array([P_init]), measurement_with_velocity, H, R, G, W, validation_gate)

# Step 6: Once we have the most likely measurement, perform the update step
if associations:
    target_index, measurement_index = associations[0]
    y = measurement_with_velocity[measurement_index]
    x_pred, P_pred = kalman_update(x_pred, P_init, y, H, R, G, W)

# Print Filtered State
print("Filtered State (fx, fy, fz, fvx, fvy, fvz):", x_pred)
