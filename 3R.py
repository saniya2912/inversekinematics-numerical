
# import numpy as np
# import matplotlib.pyplot as plt

# # Forward kinematics function
# def fwd_kin3(q):
#     l1, l2, l3 = 1, 1, 1
#     x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]) + l3 * np.cos(q[0] + q[1] + q[2])
#     y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]) + l3 * np.sin(q[0] + q[1] + q[2])
#     return np.array([x, y])

# # Cost function: Euclidean distance between the current end-effector position and the target
# def cost_function(q, target):
#     end_effector_pos = fwd_kin3(q)
#     return np.linalg.norm(target - end_effector_pos)

# # Gradient of the cost function with respect to joint angles (numerical gradient)
# def gradient(q, target, delta=1e-5):
#     grad = np.zeros_like(q)
#     for i in range(len(q)):
#         q_temp = np.copy(q)
#         q_temp[i] += delta
#         grad[i] = (cost_function(q_temp, target) - cost_function(q, target)) / delta
#     return grad

# # Gradient descent with momentum to reduce oscillations
# def inverse_kinematics(target, max_iters=500, learning_rate=0.005, momentum=0.8):
#     q = np.random.rand(3) * np.pi  # Random initial joint angles
#     velocity = np.zeros_like(q)  # Initialize velocity for momentum
#     history = [q.copy()]  # Store joint configurations for animation

#     for i in range(max_iters):
#         grad = gradient(q, target)

#         # Update velocity with momentum
#         velocity = momentum * velocity - learning_rate * grad
#         q += velocity  # Update joint angles

#         history.append(q.copy())  # Save current configuration

#         # Terminate if the distance to the target is small enough
#         if cost_function(q, target) < 1e-4:
#             break

#     return history

# # Animation function
# def animate_3r(history, target):
#     l1, l2, l3 = 1, 1, 1
#     plt.figure(figsize=(8, 8))

#     for q in history:
#         # Clear the plot for the next frame
#         plt.clf()

#         # Compute joint positions using forward kinematics
#         P0 = np.array([0, 0])
#         P1 = l1 * np.array([np.cos(q[0]), np.sin(q[0])])
#         P2 = P1 + l2 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
#         P3 = P2 + l3 * np.array([np.cos(q[0] + q[1] + q[2]), np.sin(q[0] + q[1] + q[2])])

#         # Plot manipulator arms
#         plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'g', linewidth=2)
#         plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'b', linewidth=2)
#         plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'r', linewidth=2)

#         # Plot joint positions
#         plt.plot(P0[0], P0[1], 'ok')
#         plt.plot(P1[0], P1[1], 'ok')
#         plt.plot(P2[0], P2[1], 'ok')
#         plt.plot(P3[0], P3[1], 'ok')

#         # Plot target position
#         plt.plot(target[0], target[1], 'xr', markersize=10)

#         plt.xlim([-3.5, 3.5])
#         plt.ylim([-3.5, 3.5])
#         plt.grid(True)
#         plt.xlabel('X axis (m)')
#         plt.ylabel('Y axis (m)')
#         plt.axis('equal')

#         # Pause for a brief moment to create animation effect
#         plt.pause(0.02)
#     plt.close()

#     plt.show()

# # Target end-effector position
# target = np.array([2.5, 1.5])

# # Run inverse kinematics with momentum to get the history of joint angles
# history = inverse_kinematics(target)

# # Animate the manipulator from initial to final position
# animate_3r(history, target)

import numpy as np
import matplotlib.pyplot as plt

# Forward kinematics function
def fwd_kin3(q):
    l1, l2, l3 = 1, 1, 1
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]) + l3 * np.cos(q[0] + q[1] + q[2])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]) + l3 * np.sin(q[0] + q[1] + q[2])
    return np.array([x, y])

# Cost function: Euclidean distance between the current end-effector position and the target
def cost_function(q, target):
    end_effector_pos = fwd_kin3(q)
    return np.linalg.norm(target - end_effector_pos)

# Gradient of the cost function with respect to joint angles (numerical gradient)
def gradient(q, target, delta=1e-5):
    grad = np.zeros_like(q)
    for i in range(len(q)):
        q_temp = np.copy(q)
        q_temp[i] += delta
        grad[i] = (cost_function(q_temp, target) - cost_function(q, target)) / delta
    return grad

# Simple gradient descent-based inverse kinematics
def inverse_kinematics(target, max_iters=500, learning_rate=0.01):
    q = np.random.rand(3) * np.pi  # Random initial joint angles
    history = [q.copy()]  # Store joint configurations for animation

    for i in range(max_iters):
        grad = gradient(q, target)  # Calculate gradient
        q -= learning_rate * grad  # Update joint angles
        history.append(q.copy())  # Save current configuration

        # Terminate if the distance to the target is small enough
        if cost_function(q, target) < 1e-3:
            break

    return history

# Animation function
def animate_3r(history, target):
    l1, l2, l3 = 1, 1, 1
    plt.figure(figsize=(8, 8))

    for q in history:
        # Clear the plot for the next frame
        plt.clf()

        # Compute joint positions using forward kinematics
        P0 = np.array([0, 0])
        P1 = l1 * np.array([np.cos(q[0]), np.sin(q[0])])
        P2 = P1 + l2 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
        P3 = P2 + l3 * np.array([np.cos(q[0] + q[1] + q[2]), np.sin(q[0] + q[1] + q[2])])

        # Plot manipulator arms
        plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'g', linewidth=2)
        plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'b', linewidth=2)
        plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'r', linewidth=2)

        # Plot joint positions
        plt.plot(P0[0], P0[1], 'ok')
        plt.plot(P1[0], P1[1], 'ok')
        plt.plot(P2[0], P2[1], 'ok')
        plt.plot(P3[0], P3[1], 'ok')

        # Plot target position
        plt.plot(target[0], target[1], 'xr', markersize=10)

        plt.xlim([-3.5, 3.5])
        plt.ylim([-3.5, 3.5])
        plt.grid(True)
        plt.xlabel('X axis (m)')
        plt.ylabel('Y axis (m)')
        plt.axis('equal')

        # Pause for a brief moment to create animation effect
        plt.pause(0.02)

    plt.close()
    plt.show()

# Target end-effector position
target = np.array([2.5, 1.5])

# Run inverse kinematics to get the history of joint angles
history = inverse_kinematics(target)

# Animate the manipulator from initial to final position
animate_3r(history, target)
