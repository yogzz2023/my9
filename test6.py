import numpy as np
import matplotlib.pyplot as plt
import csv

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((9, 1))  # Filter state vector with additional force components
        self.pf = np.eye(9)  # Filter state covariance matrix with additional force components
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 9)  # Measurement matrix updated to include force components
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.mass = 10  # Assumed mass value (update as needed)

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz], [0], [0], [0]])  # Initialize forces as 0
        self.Meas_Time = time

    def Filter_state_covariance(self, measurements, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(9)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(9) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

        # Update step with JPDA
        Z = np.array(measurements)
        H = np.eye(3, 9)
        Inn = Z[:, :, np.newaxis] - np.dot(H, self.Sf)
        S = np.dot(H, np.dot(self.pf, H.T)) + self.R
        K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.sum(K[:, :, np.newaxis] * Inn, axis=1)
        self.pf = np.dot(np.eye(9) - np.dot(K, H), self.pf)

        # Calculate association probabilities using JPDA
        association_probs = self.calculate_association_probabilities(Z)

        # Find the most likely associated measurement for the track with the highest marginal probability
        max_associated_index = np.argmax(association_probs)
        most_likely_associated_measurement = measurements[max_associated_index]

        # Calculate forces using Newton's second law
        ax = (self.Sf[3][0] - self.Sf[6][0]) / dt  # Calculate acceleration in x-direction
        ay = (self.Sf[4][0] - self.Sf[7][0]) / dt  # Calculate acceleration in y-direction
        az = (self.Sf[5][0] - self.Sf[8][0]) / dt  # Calculate acceleration in z-direction
        force_x = self.mass * ax  # Calculate force in x-direction
        force_y = self.mass * ay  # Calculate force in y-direction
        force_z = self.mass * az  # Calculate force in z-direction

        # Update filtered state components with calculated forces
        self.Sf[6][0] = force_x
        self.Sf[7][0] = force_y
        self.Sf[8][0] = force_z

        # Print filtered state
        print("Filtered State Components:")
        print("Filtered fx:", self.Sf[0][0])
        print("Filtered fy:", self.Sf[1][0])
        print("Filtered fz:", self.Sf[2][0])
        print("Filtered fvx:", self.Sf[3][0])
        print("Filtered fvy:", self.Sf[4][0])
        print("Filtered fvz:", self.Sf[5][0])
        print("Filtered force_x:", self.Sf[6][0])
        print("Filtered force_y:", self.Sf[7][0])
        print("Filtered force_z:", self.Sf[8][0])
        print("Filtered Time:", current_time)
        print("Most Likely Associated Measurement:", most_likely_associated_measurement)

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf, most_likely_associated_measurement

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            measurements.append((mr, ma, me, mt))
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
    filtered_state, most_likely_associated_measurement = kalman_filter.Filter_state_covariance([predicted_measurements], predicted_time)

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
