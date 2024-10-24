import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Manipulator link lengths
L1 = 0.17  # length of link1
L2 = 0.20  # length of link2
L3 = 0.20  # length of link3
L4 = 0.20  # length of link4

# Initial joint angles
initial_guess = np.array([0, 0, 0, 0])
def computeTransformationMatrix(q1, q2, q3, q4):
    '''
    takes in the current joint angles(rad) and return
    rotation matrix and displacement vector for 4DOF arm
    '''

    # # changing from degrees to radian

    # t1 = t1*math.pi/180
    # t2 = t2*math.pi/180
    # t3 = t3*math.pi/180
    # t4 = t4*math.pi/180

    # forward kinematics equations

    x = -np.cos(q1)*(L3*np.sin(q2+q3)+L2*np.sin(q2)-L4*np.cos(q2+q3+q4))
    y = -np.sin(q1)*(L3*np.sin(q2+q3)+L2 * np.sin(q2)-L4*np.cos(q2+q3+q4))
    z = L1+L3*np.cos(q2+q3)+L2*np.cos(q2)+L4*np.sin(q2+q3+q4)

    H = np.array([[np.cos(q1)*np.cos(q2+q3+q4), np.cos(q1)*-np.sin(q2+q3+q4), np.sin(q1), x],
                  [np.sin(q1)*np.cos(q2+q3+q4), np.sin(q1)*-
                   np.sin(q2+q3+q4), -np.cos(q1), y],
                  [np.sin(q2+q3+q4), np.cos(q2+q3+q4), 0, z],
                  [0, 0, 0, 1]])

    RotMat = np.matrix(np.ones((3, 3)))
    DisVector = np.matrix(np.ones((3, 1)))
    DisVector[:3, :] = H[:3, 3:]
    RotMat[:3, :3] = H[:3, :3]

    return RotMat.round(decimals=3), DisVector.round(decimals=3)

def desiredTransformationMatrix(x_desired, y_desired, z_desired):
    '''
    takes in desired position of end effector.
    desired rotation matrix defined already inside the function.
    '''
    # RotMat = np.matrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    RotMat = np.matrix(
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    DisVector = np.matrix(np.array([[x_desired], [y_desired], [z_desired]]))

    return RotMat, DisVector

def pseudoJac(t1, t2, t3, t4):
    '''
    takes in the joint angles in radian and return
    pseudo inverse of jacobian matrix.
    '''

    # the first 3 rows across the columns of this jacobian matrix J are correct.
    # confused about the last 3 rows.
    J = np.matrix([[np.sin(t1)*(L3*np.sin(t2+t3)+L2*np.sin(t2)-L4*np.cos(t2+t3+t4)),
                    -np.cos(t1)*(L3*np.cos(t2+t3)+L2 *
                                 np.cos(t2)+L4*np.sin(t2+t3+t4)),
                    -np.cos(t1)*(L3*np.cos(t2+t3)+L4*np.sin(t2+t3+t4)),
                    -L4*np.sin(t2+t3+t4)*np.cos(t1)],
                   [-np.cos(t1)*(L3*np.sin(t2+t3)+L2*np.sin(t2)-L4*np.cos(t2+t3+t4)),
                    -np.sin(t1)*(L3*np.cos(t2+t3)+L2 *
                                 np.cos(t2)+L4*np.sin(t2+t3+t4)),
                    -np.sin(t1)*(L3*np.cos(t2+t3)+L4*np.sin(t2+t3+t4)),
                    -L4*np.sin(t2+t3+t4)*np.sin(t1)],
                   [0,
                    -L3*np.sin(t2+t3)-L2*np.sin(t2)+L4*np.cos(t2+t3+t4),
                    (-L3*np.sin(t2+t3)+L4*np.cos(t2+t3+t4)),
                    L4*np.cos(t2+t3+t4)],
                   [0, np.sin(t1), np.sin(t1), np.sin(t1)],
                   [0, -np.cos(t1), -np.cos(t1), -np.cos(t1)],
                   [1, 0, 0, 0]])
    return J.T

def forward_kinematics(q):
    '''
    Takes joint angles and returns the positions of each joint
    in 3D space (end-effector position).
    '''
    q1, q2, q3, q4 = q
    
    # Compute each joint position
    x0, y0, z0 = 0, 0, 0   # Base position (origin)
    z1 = L1                # Joint 1 position (along z-axis due to L1)
    
    x2 = L2 * np.cos(q1) * np.cos(q2)
    y2 = L2 * np.sin(q1) * np.cos(q2)
    z2 = L1 + L2 * np.sin(q2)
    
    x3 = x2 + L3 * np.cos(q1) * np.cos(q2 + q3)
    y3 = y2 + L3 * np.sin(q1) * np.cos(q2 + q3)
    z3 = z2 + L3 * np.sin(q2 + q3)
    
    x4 = x3 + L4 * np.cos(q1) * np.cos(q2 + q3 + q4)
    y4 = y3 + L4 * np.sin(q1) * np.cos(q2 + q3 + q4)
    z4 = z3 + L4 * np.sin(q2 + q3 + q4)

    return np.array([[x0, y0, z0], [0, 0, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]])

def plot_manipulator(joint_positions, ax):
    '''
    Plots the manipulator links based on the joint positions
    '''
    ax.cla()  # Clear the previous plot
    ax.plot([0], [0], [0], 'bo')  # Base position
    
    # Plot each link
    for i in range(len(joint_positions) - 1):
        ax.plot([joint_positions[i][0], joint_positions[i+1][0]],
                [joint_positions[i][1], joint_positions[i+1][1]],
                [joint_positions[i][2], joint_positions[i+1][2]], 'ro-', lw=4)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def invOptWithVisualization(q1_initial, q2_initial, q3_initial, q4_initial, x, y, z):
    '''
    Perform inverse kinematics with visualization
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    errorList = np.array([1])
    err = 1
    q = np.matrix([[np.deg2rad(q1_initial)], [np.deg2rad(q2_initial)], [
                  np.deg2rad(q3_initial)], [np.deg2rad(q4_initial)]])
    
    delx = np.matrix(np.zeros((6, 1)))
    iter = 0
    Rd, Dd = desiredTransformationMatrix(x, y, z)

    jb = [
        (np.deg2rad(-180), np.deg2rad(180)),  # Bounds for q1
        (np.deg2rad(-72), np.deg2rad(72)),    # Bounds for q2
        (np.deg2rad(-150), np.deg2rad(150)),  # Bounds for q3
        (np.deg2rad(-150), np.deg2rad(150))   # Bounds for q4
    ]

    while err >= 1e-2 and iter < 250:
        Rk, Dk = computeTransformationMatrix(q[0, 0], q[1, 0], q[2, 0], q[3, 0])
        ep = Dd-Dk
        roll = np.arctan2(Rk[2, 1], Rk[2, 2])
        yaw = np.arctan2(Rk[1, 0], Rk[0, 0])

        if (np.cos(yaw) == 0):
            pitch = np.arctan2(-Rk[2, 0], (Rk[1, 0]/np.sin(yaw)))
        else:
            pitch = np.arctan2(-Rk[2, 0], (Rk[0, 0]/np.cos(yaw)))

        delx[0:3, :] = ep
        delx[3:6, :] = [[0], [np.deg2rad(5)-pitch], [0]]

        Jinv = pseudoJac(q[0, 0], q[1, 0], q[2, 0], q[3, 0])
        delq = Jinv * delx
        q = q + 0.05 * delq

        for i in range(len(jb)):
            q[i, 0] = np.clip(q[i, 0], *jb[i])

        err = np.linalg.norm(delq)
        errorList = np.append(errorList, err)

        # Get joint positions and plot the manipulator
        joint_positions = forward_kinematics(np.array(q).flatten())
        plot_manipulator(joint_positions, ax)
        
        plt.pause(0.1)
        
        iter += 1

    return np.rad2deg(q).round(decimals=1), errorList


optimizedJointAngles, errorList = invOptWithVisualization(
    initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3],
    0.58, -0.58, 0.1)

plt.figure()
plt.plot(errorList, linewidth=4, label='Position Error')
plt.xlabel('timesteps')
plt.ylabel('Error magnitude')
plt.legend()
plt.show()
