"""
Class-based code for the MuJoCo TidyBot simulation.
This should be run with mjpython: mjpython simulator.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
from simulator_util import MujocoSimulatorUtil


class MujocoSimulator:
    def __init__(self, model_path):
        """
        Initializes the simulator.
        :param model_path: Path to the MuJoCo model XML file.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path) # MuJoCo model
        self.data = mujoco.MjData(self.model) # MuJoCo data
        self.viewer = None
        self.util = MujocoSimulatorUtil(self.model)

    def run(self):
        """
        Runs the simulation loop.
        """
        if not self.viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # disable Rangefinder rendering
        
        while self.viewer.is_running():
            step_start = time.time()
            
            # Advance the physics simulation one step.
            mujoco.mj_step(self.model, self.data)
            
            # Synchronize the viewer (60 FPS).
            self.viewer.sync()
            
            # Maintain real-time speed.
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        print("Simulation ended.")
    
    def set_cube_position(self):
        """
        Sets the position (x, y) and z-axis rotation of each cube to random values.
        """
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith('cube_'):
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                
                # Get the address of the free joint in qpos
                jnt_adr = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[jnt_adr]

                # Set random x, y position, keep z the same
                random_x = np.random.uniform(-4.5, 4.5)
                random_y = np.random.uniform(-4.5, 4.5)
                self.data.qpos[qpos_adr:qpos_adr+2] = [random_x, random_y]

                # Set random z-axis rotation
                random_z_angle = np.random.uniform(0, 2 * np.pi)
                # Create quaternion from euler angle
                r = Rotation.from_euler('z', random_z_angle)
                # MuJoCo uses [w, x, y, z], SciPy uses [x, y, z, w]
                quat = r.as_quat() # [x, y, z, w]
                mujoco_quat = [quat[3], quat[0], quat[1], quat[2]]
                self.data.qpos[qpos_adr+3:qpos_adr+7] = mujoco_quat

    def get_cube_position(self):
        """
        Gets the list of cubes and their information from the simulation.
        """
        cube_list = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith('cube_'):
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                pos = self.data.body(body_id).xpos
                quat = self.data.body(body_id).xquat

                # Reorder quat from [w, x, y, z] to [x, y, z, w] for SciPy
                scipy_quat = quat[[1, 2, 3, 0]]
                r = Rotation.from_quat(scipy_quat)
                euler = r.as_euler('xyz', degrees=False)
                
                cube_info = {
                    'name': body_name,
                    'position': np.round(pos, 2).tolist(),
                    'euler': np.round(euler, 2).tolist()
                }
                cube_list.append(cube_info)
        return cube_list
    
    def move_to(self, target):
        """
        Moves the robot's base towards a target position, stopping 50cm short and orienting to face the target.
        """
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target)
        target_pos = self.data.body(target_id).xpos

        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "joint_0_site")
        base_pos = self.data.site(base_id).xpos

        direction = target_pos - base_pos
        distance = np.linalg.norm(direction)

        # Calculate and set yaw angle to face the target
        theta_z = np.arctan2(direction[1], direction[0])
        self.data.ctrl[2] = theta_z

        # if we are closer than 50cm, just face the target and don't move
        if distance < 0.5:
            return

        # Calculate the new position 50cm before the target
        unit_direction = direction / distance
        new_target_pos = target_pos - 0.5 * unit_direction

        # Set position control
        self.data.ctrl[0] = new_target_pos[0]
        self.data.ctrl[1] = new_target_pos[1]
    
    def pick_up(self, target):
        """
        Pick up the target.
        """
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target)
        target_pos = self.data.body(target_id).xpos.copy()
        target_pos[2] += 0.1
        target_quat = self.data.body(target_id).xquat.copy()

        scipy_quat = target_quat[[1, 2, 3, 0]]
        r = Rotation.from_quat(scipy_quat)
        target_rpy = r.as_euler('xyz', degrees=False)

        joints = self.util.solve_ik(target_pos, target_rpy, self.data)
        self.data.ctrl[3:10] = joints


def main():
    """
    Main function.
    """
    simulator = MujocoSimulator()
    simulator.run()


if __name__ == "__main__":
    main()