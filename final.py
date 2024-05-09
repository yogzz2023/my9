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

    def Predict_Measurement(self, vel_x, vel_y, vel_z, dt):
        # Check the shape of self.Sf
        if self.Sf.shape != (6, 1):
            raise ValueError("Invalid shape for self.Sf: expected (6, 1)")

        # Predict next measurement
        predicted_time = self.Meas_Time + dt
        predicted_measurement = (self.Sf[0] + vel_x * dt,
                                 self.Sf[1] + vel_y * dt,
                                 self.Sf[2] + vel_z * dt)
        return predicted_measurement, predicted_time

    def Update_Filter(self, measurements, current_time):
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
        association_probs = []
        associated_measurements = []

        for measurement in measurements:
            Z = np.array([measurement])
            H = np.eye(3, 6)
            Inn = Z - np.dot(H, self.Sf)
            S = np.dot(H, np.dot(self.pf, H.T)) + self.R
            K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
            updated_state = self.Sf + np.dot(K, Inn.T).T
            innovation_covariance = np.dot(np.dot(H, self.pf), H.T)
            det_S = np.linalg.det(S)

            # Handle the overflow error in the exponential function
            try:
                mahalanobis_distance_squared = np.sum(np.dot(Inn.transpose(0, 2, 1), np.linalg.inv(S)) * Inn, axis=1)
                association_prob = 1 / ((2 * np.pi) ** (3 / 2) * np.sqrt(det_S)) * np.exp(-0.5 * mahalanobis_distance_squared)
            except FloatingPointError:
                association_prob = 0

            association_probs.append(association_prob)
            associated_measurements.append(updated_state)

        # Find the most likely associated measurement for the track with the highest marginal probability
        if associated_measurements:
            max_associated_index = np.argmax(np.sum(association_probs, axis=0))
            if max_associated_index < len(associated_measurements):
                most_likely_associated_measurement = associated_measurements[max_associated_index]
            else:
                most_likely_associated_measurement = self.Sf
        else:
            most_likely_associated_measurement = self.Sf

        # Update filter with the most likely associated measurement
        self.Sf = most_likely_associated_measurement
        self.pf = self.pf - np.dot(K, np.dot(H, self.pf))

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # Measurement range
            ma = float(row[11])  # Measurement azimuth
            me = float(row[12])  # Measurement elevation
            mt = float(row[13])  # Measurement time
            measurements.append((mr, ma, me, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_57.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Perform steps for each measurement
for i in range(len(measurements)):
    # Step 1: Initialize with measurement M1 then initialize measurement M2
    if i == 0:
        m1 = measurements[i]
        kalman_filter.Initialize_Filter_state_covariance(m1[0], m1[1], m1[2], 0, 0, 0, m1[3])
    elif i == 1:
        m2 = measurements[i]
        kalman_filter.Initialize_Filter_state_covariance(m2[0], m2[1], m2[2], 0, 0, 0, m2[3])
    else:
        # Step 2: Predict Measurement M3
        prev_m = measurements[i - 1]
        prev_prev_m = measurements[i - 2]
        
        # Calculate velocities
        vel_x = (prev_m[0] - prev_prev_m[0]) / (prev_m[3] - prev_prev_m[3])
        vel_y = (prev_m[1] - prev_prev_m[1]) / (prev_m[3] - prev_prev_m[3])
        vel_z = (prev_m[2] - prev_prev_m[2]) / (prev_m[3] - prev_prev_m[3])
        
        # Predict Measurement M3
        predicted_measurement, predicted_time = kalman_filter.Predict_Measurement(vel_x, vel_y, vel_z, prev_m[3] - prev_prev_m[3])

        # Step 3: Perform Joint Probabilistic Data Association
        # (Currently implemented in Update_Filter method)

        # Step 4: Associate measurements with the predicted state at time M2
        kalman_filter.Update_Filter([predicted_measurement], predicted_time)

# Print final filtered state
print("Filtered State Components:")
print("Filtered fx:", kalman_filter.Sf[0][0])
print("Filtered fy:", kalman_filter.Sf[1][0])
print("Filtered fz:", kalman_filter.Sf[2][0])
print("Filtered fvx:", kalman_filter.Sf[3][0])
print("Filtered fvy:", kalman_filter.Sf[4][0])
print("Filtered fvz:", kalman_filter.Sf[5][0])
print("Filtered Time:", kalman_filter.Meas_Time)
