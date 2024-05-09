import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
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

    def Filter_state_covariance(self, measurements, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

        # Update step
        Z = np.array(measurements)
        H = np.eye(3, 6)
        Inn = Z - np.dot(H, self.Sf)
        S = np.dot(H, np.dot(self.pf, H.T)) + self.R
        K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)

        # Print filtered state components
        print("Filtered State Components:")
        print("Filtered fx:", self.Sf[0][0])
        print("Filtered fy:", self.Sf[1][0])
        print("Filtered fz:", self.Sf[2][0])
        print("Filtered fvx:", self.Sf[3][0])
        print("Filtered fvy:", self.Sf[4][0])
        print("Filtered fvz:", self.Sf[5][0])
        print("Filtered Time:", current_time)

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf, measurements

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            rng1 = float(row[10])  # Measurement range
            az = float(row[11])    # Measurement azimuth
            el = float(row[12])    # Measurement elevation
            time = float(row[13]) # Measurement time
            measurements.append((rng1, az, el, time))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_57.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Step 1: Initialize with 1st measurement M1
kalman_filter.Initialize_Filter_state_covariance(measurements[0][0], measurements[0][1], measurements[0][2], 0, 0, 0, measurements[0][3])

# Step 2: Initialize with 2nd measurement M2
kalman_filter.Initialize_Filter_state_covariance(measurements[1][0], measurements[1][1], measurements[1][2], 0, 0, 0, measurements[1][3])

# Lists to store predicted values
predicted_range = []
predicted_azimuth = []
predicted_elevation = []

# Process measurements and get predicted state estimates at each time step
for i in range(2, len(measurements)):
    # Step 3: Get the velocity from step 1 and 2
    vel_x = (measurements[1][0] - measurements[0][0]) / (measurements[1][3] - measurements[0][3])
    vel_y = (measurements[1][1] - measurements[0][1]) / (measurements[1][3] - measurements[0][3])
    vel_z = (measurements[1][2] - measurements[0][2]) / (measurements[1][3] - measurements[0][3])

    # Step 4: Get measurement M3
    predicted_time = measurements[i][3]
    predicted_measurements = (measurements[i][0] + vel_x * (predicted_time - measurements[1][3]), 
                               measurements[i][1] + vel_y * (predicted_time - measurements[1][3]), 
                               measurements[i][2] + vel_z * (predicted_time - measurements[1][3]))
    
    # Step 5: Do association using JPDA
    filtered_state, most_likely_measurement = kalman_filter.Filter_state_covariance([predicted_measurements], predicted_time)

    # Append predicted values to lists
    predicted_range.append(filtered_state[0][0])
    predicted_azimuth.append(filtered_state[1][0])
    predicted_elevation.append(filtered_state[2][0])

# Plotting predicted values
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(predicted_range, label='Predicted Range')
plt.xlabel('Time Step')
plt.ylabel('Range')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(predicted_azimuth, label='Predicted Azimuth')
plt.xlabel('Time Step')
plt.ylabel('Azimuth')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(predicted_elevation, label='Predicted Elevation')
plt.xlabel('Time Step')
plt.ylabel('Elevation')
plt.legend()

plt.tight_layout()
plt.show()
