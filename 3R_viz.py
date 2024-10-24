import numpy as np
import matplotlib.pyplot as plt

# Time parameters
dt = 1 / 1000
t = np.arange(0, 12 + dt, dt)

# Scale and trajectory definition
scale = 2 / (3 - np.cos(2 * t))
x = scale * np.cos(t) * 3
y = scale * np.sin(2 * t) / 2 * 3

# Velocity calculation
dx = np.diff(x) / dt
dy = np.diff(y) / dt
v = np.vstack((dx, dy))

# Jacobian and forward kinematics
def jaco_3(th1, th2, th3):
    l1, l2, l3 = 1, 1, 1
    J = np.array([
        [-l1 * np.sin(th1) - l2 * np.sin(th2 + th1) - l3 * np.sin(th3 + th2 + th1),
         -l2 * np.sin(th2 + th1) - l3 * np.sin(th3 + th2 + th1),
         -l3 * np.sin(th1 + th2 + th3)],
        [l1 * np.cos(th1) + l2 * np.cos(th2 + th1) + l3 * np.cos(th3 + th2 + th1),
         l2 * np.cos(th2 + th1) + l3 * np.cos(th3 + th2 + th1),
         l3 * np.cos(th1 + th2 + th3)]
    ])
    return J

def fwd_kin3(q):
    l1, l2, l3 = 1, 1, 1
    x = l1 * np.cos(q[0, :]) + l2 * np.cos(q[0, :] + q[1, :]) + l3 * np.cos(q[0, :] + q[1, :] + q[2, :])
    y = l1 * np.sin(q[0, :]) + l2 * np.sin(q[0, :] + q[1, :]) + l3 * np.sin(q[0, :] + q[1, :] + q[2, :])
    return x, y

def animate_3r(q, K, xt, yt):
    plt.figure(figsize=(10, 10))
    c = 0
    for i in range(0, len(q[0]), 15):
        theta1 = q[0, i]
        theta2 = q[1, i]
        theta3 = q[2, i]
        
        l1, l2, l3 = 1, 1, 1
        
        # Homogeneous transformation matrices
        H01 = np.array([[np.cos(theta1), -np.sin(theta1), 0, l1 * np.cos(theta1)],
                        [np.sin(theta1), np.cos(theta1), 0, l1 * np.sin(theta1)],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        H12 = np.array([[np.cos(theta2), -np.sin(theta2), 0, l2 * np.cos(theta2)],
                        [np.sin(theta2), np.cos(theta2), 0, l2 * np.sin(theta2)],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        H23 = np.array([[np.cos(theta3), -np.sin(theta3), 0, l3 * np.cos(theta3)],
                        [np.sin(theta3), np.cos(theta3), 0, l3 * np.sin(theta3)],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        H02 = np.dot(H01, H12)
        H03 = np.dot(np.dot(H01, H12), H23)

        P1 = H01[:2, 3]
        P2 = H02[:2, 3]
        P3 = H03[:2, 3]

        plt.plot(xt, yt, '--y')
        plt.plot(P1[0], P1[1], 'ok', linewidth=1)
        plt.plot([0, P1[0]], [0, P1[1]], 'g', linewidth=2)
        plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'b', linewidth=2)
        plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'g', linewidth=2)
        plt.plot(P2[0], P2[1], 'ok', linewidth=1)
        plt.plot(P3[0], P3[1], 'ok', linewidth=1)

        plt.xlim([-3.5, 3.5])
        plt.ylim([-3.5, 3.5])
        plt.axis('square')
        plt.grid(True)
        plt.xlabel('X axis (m)', fontsize=18)
        plt.ylabel('Y axis (m)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.pause(0.05)
        plt.clf()  # Clear figure for the next frame
        c += 1

# Main loop to compute joint angles
q = np.zeros((3, len(x)))
ath = 0
K = np.array([[np.cos(np.radians(ath)), 0],
              [np.sin(np.radians(ath)), 0]])

# Initial joint angles for the first case
q[:, 0] = [np.pi / 32, 0, 0]

for k in range(len(x) - 1):
    th1, th2, th3 = q[:, k]
    J = jaco_3(th1, th2, th3)
    q[:, k + 1] = q[:, k] + np.linalg.pinv(J) @ v[:, k] * dt

animate_3r(q, K, x, y)

# Example for a different initial configuration
q = np.zeros((3, len(x)))
q[:, 0] = [np.pi / 32, 1, 0]

for k in range(len(x) - 1):
    th1, th2, th3 = q[:, k]
    J = jaco_3(th1, th2, th3)
    x0, y0 = fwd_kin3(q[:, k])
    E = np.array([x[k], y[k]]) - np.array([x0, y0])
    q[:, k + 1] = q[:, k] + np.linalg.pinv(J) @ (v[:, k] + 1E2 * E) * dt

animate_3r(q, K, x, y)
