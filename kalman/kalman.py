from pykalman import UnscentedKalmanFilter
import numpy as np


def update_position(measured_position, predicted_position):
        """
        Update the object's position, using the measured position or predicted position based on the error threshold.
        
        Parameters:
            measured_position (tuple or None): The measured position (x, y) or None if the measurement failed.

        Returns:
            tuple: The new position of the object.
        """
        error_threshold=500.0
        
        # if last_position is None:
        #     # This is the first measurement received.
        #     if measured_position is not None:
        #         last_position = np.array(measured_position)
        #     return tuple(last_position)

        if measured_position is None:
            # If no measurement is received, use the predicted position
            new_position = predicted_position
        else:
            # Calculate the error between the predicted and measured positions
            error = np.linalg.norm(np.array(predicted_position) - np.array(measured_position))

            print(error)

            if error > error_threshold:
                # If error is too high, trust the prediction over the measurement
                new_position = predicted_position
            else:
                # Update velocity based on the new measured position
                # self.velocity = np.array(measured_position) - self.last_position
                new_position = np.array(measured_position)

        # Update the last known position
        # last_position = new_position
        return new_position


def state_transition_function(state, noise):
    # Example assumes next state is previous state plus some process noise
    # Adjust as necessary to match the object's dynamics
    return np.array([state[0] + state[2], state[1] + state[3], state[2], state[3]]) + noise

def observation_function(state, noise):
    # We only observe positions, not velocities
    return np.array([state[0], state[1]]) + noise

# Initial conditions
initial_state_mean = [0, 0, 0, 0]  # Start at (0,0) with no velocity
initial_state_covariance = 0.1 * np.eye(4)  # Initial uncertainty

# Noise covariances
transition_covariance = 0.01 * np.eye(4)  # Process noise
observation_covariance = 0.1 * np.eye(2)  # Observation noise

# Instantiate the Unscented Kalman Filter
ukf = UnscentedKalmanFilter(
    transition_functions=state_transition_function,
    observation_functions=observation_function,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance
)

def predict_next_position(measurements):
    # Update UKF with all current measurements and predict the next state
    # Smooth to get all states, then use the last state to predict the next one
    smoothed_states_means, smoothed_states_covariances = ukf.smooth(measurements)
    last_state_mean = smoothed_states_means[-1]
    last_state_covariance = smoothed_states_covariances[-1]
    next_state_mean, next_state_covariance = ukf.filter_update(
        last_state_mean, last_state_covariance
    )
    return next_state_mean[:2]  # return predicted position (x, y)

# Example usage
# measurements = np.array([
#     [1, 2],  # Sample measurements
#     [2, 3],
#     [3, 4]
# ])

# measurements = [
#     [1, 2],  # Sample measurements
#     [2, 3],
#     [3, 4]
# ]

measurements = [
    (1, 2),  # Sample measurements
    (2, 3),
    (3, 4)
]


next_position = predict_next_position(measurements)
# print("Predicted next position:", next_position)
