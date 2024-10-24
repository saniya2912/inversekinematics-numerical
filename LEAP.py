import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media
import time
from main_2sept import *


class GradientDescentIK:
    def __init__(self,xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.jacp = np.zeros((3, self.model.nv))  # translation jacobian
        self.jacr = np.zeros((3, self.model.nv)) 
        self.step_size = 0.5
        self.tol = 0.01
        self.alpha = 0.5
        self.init_q = [0.0, 0.0, 0.0, 0.0]  
    
    def check_joint_limits(self, q):
        """Check if the joints are under or above their limits."""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))


    def calculate(self, goal_pos, goal_rot, site_name):
        self.data.qpos = self.init_q
        mujoco.mj_forward(self.model, self.data)

        site_id= self.model.site(site_name).id
        
        # Current pose and orientation
        current_pos = self.data.site(site_id).xpos
        current_rot = self.data.site(site_id).xmat.reshape(3, 3)

        # Position and orientation error
        pos_error = np.subtract(goal_pos, current_pos)
        rot_error = 0.5 * (np.cross(current_rot[:, 0], goal_rot[:, 0]) +
                           np.cross(current_rot[:, 1], goal_rot[:, 1]) +
                           np.cross(current_rot[:, 2], goal_rot[:, 2]))

        # Combine position and orientation errors
        error = np.concatenate([pos_error, rot_error])

        max_iterations = 100000
        iteration = 0

        while np.linalg.norm(error) >= self.tol and iteration < max_iterations:
            # Calculate Jacobian for position and orientation
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, site_id)
            full_jacobian = np.vstack((self.jacp, self.jacr))
            
            # Calculate gradient
            grad = self.alpha * full_jacobian.T @ error
            
            # Compute next step
            self.data.qpos += self.step_size * grad
            
            # Check joint limits
            self.check_joint_limits(self.data.qpos)
            
            # Compute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Update position and orientation error
            current_pos = self.data.site(site_id).xpos
            current_rot = self.data.site(site_id).xmat.reshape(3, 3)
            pos_error = np.subtract(goal_pos, current_pos)
            rot_error = 0.5 * (np.cross(current_rot[:, 0], goal_rot[:, 0]) +
                               np.cross(current_rot[:, 1], goal_rot[:, 1]) +
                               np.cross(current_rot[:, 2], goal_rot[:, 2]))
            error = np.concatenate([pos_error, rot_error])

            iteration += 1

        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached. The solution may not have converged.")
        
        result = self.data.qpos.copy()
        return result
    
class OnlyPosIK:
    def __init__(self,xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.jacp = np.zeros((3, self.model.nv))  # translation jacobian
        self.jacr = np.zeros((3, self.model.nv)) 
        self.step_size = 0.5
        self.tol = 0.01
        self.alpha = 0.5
        self.init_q = [0.0, 0.0, 0.0, 0.0]  
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Gradient Descent pseudocode implementation
    def calculate(self, goal, site_name):
        site_id=self.model.site(site_name).id
        self.data.qpos = self.init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.site(site_id).xpos
        error = np.subtract(goal, current_pose)

        max_iterations = 100000
        iteration = 0

        while (np.linalg.norm(error) >= self.tol) and iteration < max_iterations:
            #calculate jacobian
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr,site_id)
            #calculate gradient
            grad = self.alpha * self.jacp.T @ error
            #compute next step
            self.data.qpos += self.step_size * grad
            #check joint limits
            self.check_joint_limits(self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            error = np.subtract(goal, self.data.site(site_id).xpos)

            iteration += 1

        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached. The solution may not have converged.")
        
        result = self.data.qpos.copy()
        return result
 # Record start time

model_path = '/home/iitgn-robotics/Saniya/redundancy-leap/leap-mujoco/model/leap hand/leaphand_redundancy.xml'
leap_hand = LeapNodeMujoco(model_path)

index_path='/home/iitgn-robotics/Saniya/inversekinematics-numerical/mujoco-3.1.6/model/leap hand/leaphand_redundancy.xml'
thumb_path='/home/iitgn-robotics/Saniya/inversekinematics-numerical/mujoco-3.1.6/model/leap hand/redundancy/0_thumb_sim.xml'

IK_index=OnlyPosIK(index_path)
IK_thumb=OnlyPosIK(thumb_path)

mujoco.mj_step(leap_hand.m,leap_hand.d)

pos_worldi1=leap_hand.d.site(leap_hand.m.site('contact_index1').id).xpos.reshape(3)
rot_worldi1=leap_hand.d.site(leap_hand.m.site('contact_index1').id).xmat.reshape(3,3)
pos_worldt1=leap_hand.d.site(leap_hand.m.site('contact_thumb1').id).xpos.reshape(3)
rot_worldt1=leap_hand.d.site(leap_hand.m.site('contact_thumb1').id).xmat.reshape(3,3)

pos_worldi2=leap_hand.d.site(leap_hand.m.site('contact_index2').id).xpos.reshape(3)
rot_worldi2=leap_hand.d.site(leap_hand.m.site('contact_index2').id).xmat.reshape(3,3)
pos_worldt2=leap_hand.d.site(leap_hand.m.site('contact_thumb2').id).xpos.reshape(3)
rot_worldt2=leap_hand.d.site(leap_hand.m.site('contact_thumb2').id).xmat.reshape(3,3)

result_index1 = IK_index.calculate(pos_worldi1, 'contact_index')
result_thumb1 = IK_thumb.calculate(pos_worldt1, 'contact_thumb_end')
result_index2 = IK_index.calculate(pos_worldi2, 'contact_index')
result_thumb2 = IK_thumb.calculate(pos_worldt2, 'contact_thumb_end')

framerate = 30  # Set video frame rate
simulation_duration = 120  # Total duration in seconds
start_time = time.time() 
# Launch the MuJoCo viewer and run the simulation
with mujoco.viewer.launch_passive(leap_hand.m, leap_hand.d) as viewer:
    while time.time() - start_time < simulation_duration:
        current_time = time.time() - start_time  # Track simulation time

        # Phase 1: Move close to the contact points (first 2 seconds)
        if 1 < current_time < 3:
            [a, b, c, d] = result_index1
            [e, f, g, h] = result_thumb1
         
            leap_hand.apply_controls_hand([a, b, c, d, 0, 0, 0, 0, 0, 0, 0, 0, e, f, g, h])
            #leap_hand.d.qpos[-16:]=[a, b, c, d, 0, 0, 0, 0, 0, 0, 0, 0, e, f, g, h]

        # Phase 2: Continue to refine the contact (2 to 4 seconds)
        elif 3 < current_time < 8:
            [a, b, c, d] = result_index2
            [e, f, g, h] = result_thumb2
            
            leap_hand.apply_controls_hand([a, b, c, d, 0, 0, 0, 0, 0, 0, 0, 0, e, f, g, h])
            #leap_hand.d.qpos[-16:]=[a, b, c, d, 0, 0, 0, 0, 0, 0, 0, 0, e, f, g, h]

            
        # Step the simulation without camera-specific logic
        mujoco.mj_step(leap_hand.m, leap_hand.d)
        viewer.sync()