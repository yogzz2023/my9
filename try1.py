import numpy as np
import matplotlib.pyplot as plt

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def Filter_state_covariance(self, measurements, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

        Z = np.array(measurements)
        H = np.eye(3, 6)
        Inn = Z[:, :, np.newaxis] - np.dot(H, self.Sf)
        S = np.dot(H, np.dot(self.pf, H.T)) + self.R
        K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.sum(K[:, :, np.newaxis] * Inn, axis=1)
        self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)

        association_probs = self.calculate_association_probabilities(Z)

        max_associated_index = np.argmax(association_probs)
        most_likely_associated_measurement = measurements[max_associated_index]

        print("Filtered State Components:")
        print("Filtered fx:", self.Sf[0][0])
        print("Filtered fy:", self.Sf[1][0])
        print("Filtered fz:", self.Sf[2][0])
        print("Filtered fvx:", self.Sf[3][0])
        print("Filtered fvy:", self.Sf[4][0])
        print("Filtered fvz:", self.Sf[5][0])
        print("Filtered Time:", current_time)
        print("Most Likely Associated Measurement:", most_likely_associated_measurement)

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf, most_likely_associated_measurement

    def calculate_association_probabilities(self, measurements):
        num_measurements = len(measurements)
        num_tracks = 1  # For simplicity, assuming only one track
        conditional_probs = np.zeros((num_measurements, num_tracks))

        for i in range(num_measurements):
            for j in range(num_tracks):
                # Calculate conditional probability using JPDA
                conditional_probs[i, j] = 1.0 / num_measurements

        marginal_probs = np.sum(conditional_probs, axis=1) / num_tracks
        joint_probs = conditional_probs * marginal_probs[:, np.newaxis]
        association_probs = np.sum(joint_probs, axis=0)

        return association_probs

# Measurement values
measurements = [
    (20665.41, 178.8938, 1.7606, 21795.857),
    (20666.14, 178.9428, 1.7239, 21796.389),
    (20666.49, 178.8373, 1.71, 21796.887),
    (20666.46, 178.9346, 1.776, 21797.367),
    (20667.39, 178.9166, 1.8053, 21797.852),
    (20679.63, 178.8026, 2.3944, 21798.961),
    (20668.63, 178.8364, 1.7196, 21799.494),
    (20679.73, 178.9656, 1.7248, 21799.996),
    (20679.9, 178.7023, 1.6897, 21800.549),
    (20681.38, 178.9606, 1.6158, 21801.08),
    (33632.25, 296.9022, 5.2176, 22252.645),
    (33713.09, 297.0009, 5.2583, 22253.18),
    (33779.16, 297.0367, 5.226, 22253.699),
    (33986.5, 297.2512, 5.1722, 22255.199),
    (34086.27, 297.2718, 4.9672, 22255.721),
    (34274.89, 297.5085, 5.0913, 22257.18),
    (34354.61, 297.5762, 4.9576, 22257.678),
    (34568.59, 297.8105, 4.8639, 22259.193),
    (34717.52, 297.9439, 4.789, 22260.213),
    (34943.71, 298.0376, 4.7376, 22261.717),
    (35140.06, 298.2941, 4.8053, 22263.217),
    (35357.11, 298.4943, 4.6953, 22264.707),
    (35598.12, 298.7462, 4.6313, 22266.199),
    (35806.11, 298.8661, 4.6102, 22267.729),
    (36025.82, 299.0423, 4.6156, 22269.189),
    (36239.5, 299.282, 4.5413, 22270.691),
    (36469.04, 299.3902, 4.5713, 22272.172),
    (36689.36, 299.584, 4.5748, 22273.68),
    (36911.89, 299.7541, 4.5876, 22275.184),
    (37141.31, 299.9243, 4.5718, 22276.734),
    (37369.89, 300.2742, 4.6584, 22278.242),
    (37587.8, 300.2986, 4.5271, 22279.756)
]

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Step 1: Initialize with 1st measurement M1
kalman_filter.Initialize_Filter_state_covariance(measurements[0][0], measurements[0][1], measurements[0][2], 0, 0, 0, measurements[0][3])

# Step 2: Initialize with 2nd measurement M2
kalman_filter.Initialize_Filter_state_covariance(measurements[1][0], measurements[1][1], measurements[1][2], 0, 0, 0, measurements[1][3])

# Lists to store predicted values
filtered_x = []
filtered_y = []
filtered_z = []
filtered_time = []

# Process measurements and get predicted state estimates at each time step
for i in range(2, len(measurements)):
    # Step 3: Get the velocity from step 1 and 2
    vel_x = (measurements[i][0] - measurements[i-1][0]) / (measurements[i][3] - measurements[i-1][3])
    vel_y = (measurements[i][1] - measurements[i-1][1]) / (measurements[i][3] - measurements[i-1][3])
    vel_z = (measurements[i][2] - measurements[i-1][2]) / (measurements[i][3] - measurements[i-1][3])

    # Step 4: Get measurement M3
    predicted_time = measurements[i][3]
    predicted_measurements = (measurements[i][0] + vel_x * (predicted_time - measurements[i-1][3]), 
                             measurements[i][1] + vel_y * (predicted_time - measurements[i-1][3]), 
                             measurements[i][2] + vel_z * (predicted_time - measurements[i-1][3]))

    # Step 5: Do association using JPDA
    filtered_state, most_likely_associated_measurement = kalman_filter.Filter_state_covariance([predicted_measurements], predicted_time)

    # Append filtered values to lists
    filtered_x.append(filtered_state[0][0])
    filtered_y.append(filtered_state[1][0])
    filtered_z.append(filtered_state[2][0])
    filtered_time.append(predicted_time)

    # Plot the measurements and the most likely associated measurement
    plt.figure(figsize=(8, 6))
    plt.plot(filtered_time[:-1], [m[0] for m in measurements[2:i]], 'bo', label='Measurements')
    plt.plot(filtered_time[i-2], most_likely_associated_measurement[0], 'ro', label='Most Likely Associated Measurement')
    plt.xlabel('Time')
    plt.ylabel('Measurement (x)')
    plt.title('Measurements and Most Likely Associated Measurement')
    plt.legend()
    plt.grid(True)
    plt.show()

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
