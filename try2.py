import numpy as np
import matplotlib.pyplot as plt
import csv

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def Filter_state_covariance(self, measurement, current_time, velocity):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

        # Update step with JPDA
        Z = np.array([measurement])
        H = np.eye(3, 6)
        Inn = Z[:, :, np.newaxis] - np.dot(H, self.Sf)
        S = np.dot(H, np.dot(self.pf, H.T)) + self.R
        K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.sum(K[:, :, np.newaxis] * Inn, axis=1)
        self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)

        # Print filtered state
        print("Filtered State Components:")
        print("Filtered fx:", self.Sf[0][0])
        print("Filtered fy:", self.Sf[1][0])
        print("Filtered fz:", self.Sf[2][0])
        print("Filtered fvx:", self.Sf[3][0])
        print("Filtered fvy:", self.Sf[4][0])
        print("Filtered fvz:", self.Sf[5][0])
        print("Filtered Time:", current_time)

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf

# Function to generate random measurements satisfying given conditions
def generate_random_measurement():
    # Randomly generate measurements satisfying conditions
    measurement_range = np.random.uniform(0, 100)
    measurement_azimuth = np.random.uniform(0, 360)
    measurement_elevation = np.random.uniform(0, 90)
    measurement_time = np.random.uniform(0, 10)
    return measurement_range, measurement_azimuth, measurement_elevation, measurement_time

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Lists to store filtered values
filtered_x = []
filtered_y = []
filtered_z = []
filtered_time = []

# Step 1: Initialize with 1st measurement M1
m1 = generate_random_measurement()
kalman_filter.Initialize_Filter_state_covariance(m1[0], m1[1], m1[2], 0, 0, 0, m1[3])

# Step 2: Initialize with 2nd measurement M2
m2 = generate_random_measurement()
kalman_filter.Initialize_Filter_state_covariance(m2[0], m2[1], m2[2], 0, 0, 0, m2[3])

# Process measurements and get predicted state estimates at each time step
for i in range(2, 10):
    # Step 3: Predict Measurement M3 using M1, M2, and their velocities
    vel_x = (m2[0] - m1[0]) / (m2[3] - m1[3])
    vel_y = (m2[1] - m1[1]) / (m2[3] - m1[3])
    vel_z = (m2[2] - m1[2]) / (m2[3] - m1[3])
    predicted_time = np.random.uniform(m2[3], m2[3] + 10)  # Predicted time for M3
    predicted_measurement = (m2[0] + vel_x * (predicted_time - m2[3]),
                             m2[1] + vel_y * (predicted_time - m2[3]),
                             m2[2] + vel_z * (predicted_time - m2[3]))

    # Step 4: Perform update step with JPDA
    filtered_state = kalman_filter.Filter_state_covariance(predicted_measurement, predicted_time, (vel_x, vel_y, vel_z))

    # Append filtered values to lists
    filtered_x.append(filtered_state[0][0])
    filtered_y.append(filtered_state[1][0])
    filtered_z.append(filtered_state[2][0])
    filtered_time.append(predicted_time)

# Plotting filtered x, y, and z components against time
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(filtered_time, filtered_x, label='Filtered X')
plt.xlabel('Time')
plt.ylabel('Filtered X')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(filtered_time, filtered_y, label='Filtered Y')
plt.xlabel('Time')
plt.ylabel('Filtered Y')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(filtered_time, filtered_z, label='Filtered Z')
plt.xlabel('Time')
plt.ylabel('Filtered Z')
plt.legend()

plt.tight_layout()
plt.show()
