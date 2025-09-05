"""
Class-based code for the MuJoCo TidyBot simulation.
This should be run with mjpython: mjpython simulator.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
from simulator_util import IKSolver


# --- Helper Functions ---

def _s_curve_interpolation(alpha):
    """Calculates S-curve (smoothstep) interpolation."""
    return 3 * alpha**2 - 2 * alpha**3

def _scipy_to_mujoco_quat(scipy_quat):
    """Convert SciPy quaternion [x, y, z, w] to MuJoCo quaternion [w, x, y, z]."""
    return np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])

def _mujoco_to_scipy_quat(mujoco_quat):
    """Convert MuJoCo quaternion [w, x, y, z] to SciPy quaternion [x, y, z, w]."""
    return np.array([mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]])


# --- Trajectory Classes ---

class Trajectory:
    """Base class for robot movement trajectories."""
    
    def __init__(self, trajectory_type, steps, description):
        self.type = trajectory_type
        self.steps = steps
        self.current_step = 0
        self.description = description


class BaseTrajectory(Trajectory):
    """Trajectory for robot base movements (position and rotation)."""
    
    def __init__(self, start_pos, target_pos, start_theta, target_theta, duration, description="Base movement"):
        steps = int(duration * 25)  # 25Hz
        super().__init__('base', steps, description)
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.start_theta = start_theta
        self.target_theta = target_theta


class GripperTrajectory(Trajectory):
    """Trajectory for gripper movements (open/close)."""
    
    def __init__(self, start_gripper, target_gripper, duration, description="Gripper movement"):
        steps = int(duration * 100)  # 100Hz
        super().__init__('gripper', steps, description)
        self.start_gripper = start_gripper
        self.target_gripper = target_gripper


class ArmTrajectory(Trajectory):
    """Trajectory for robot arm movements using inverse kinematics."""
    
    def __init__(self, target_pos, target_rpy, duration, description="Arm movement"):
        steps = int(duration * 100)  # 100Hz
        super().__init__('arm', steps, description)
        self.target_pos = target_pos
        self.target_rpy = target_rpy


class PickupSequence(Trajectory):
    """Complex trajectory for 4-phase pickup operations."""
    
    def __init__(self, target, target_pos, target_rpy):
        super().__init__('pickup_sequence', 999999, f"{target} pickup sequence")
        self.target = target
        self.trajectories = self._create_trajectories(target, target_pos, target_rpy)
        self.current_trajectory_index = 0
    
    def _create_trajectories(self, target, target_pos, target_rpy):
        """Create 4-phase pickup sequence using trajectory classes."""
        positions = {
            'approach': target_pos + np.array([0, 0, 0.1]),   # Safe approach height
            'grasp': target_pos + np.array([0, 0, 0.02]),     # Grasp position
            'lift': target_pos + np.array([0, 0, 0.2])        # Lift height
        }
        
        return [
            ArmTrajectory(positions['approach'], target_rpy, 2.0, f"Approach {target}"),
            ArmTrajectory(positions['grasp'], target_rpy, 1.5, f"Move to grasp {target}"),
            ArmTrajectory(positions['grasp'], target_rpy, 0.5, f"Move to grasp {target}"),
            GripperTrajectory(0, 250, 1.0, f"Close gripper"),
            ArmTrajectory(positions['lift'], target_rpy, 2.0, f"Lift {target}")
        ]


# --- Main Simulator Class ---

class MujocoSimulator:
    """Main simulator class handling robot control and movement trajectories."""
    
    def __init__(self, model_path):
        """Initialize the simulator with robot model."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.util = IKSolver(self.model)
        
        self.movement_queue = []
        self.current_trajectory = None
        self.last_trajectory_time = 0.0

    def run(self):
        """Run the simulation loop."""
        if not self.viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        
        while self.viewer.is_running():
            step_start = time.time()
            
            self._update_trajectory_queue()
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        print("Simulation terminated.")

    # --- Public Methods: Robot Actions ---

    def move_to(self, target_name):
        """Move robot base toward a target object, stopping 50cm away."""
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        target_pos = self.data.body(target_id).xpos
        
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "joint_0_site")
        base_pos = self.data.site(base_id).xpos
        
        direction = target_pos - base_pos
        distance = np.linalg.norm(direction)
        theta_z = np.arctan2(direction[1], direction[0])
        
        if distance < 0.5:
            # Just rotate to face target if already close
            current_pos = base_pos[:2].copy()
            current_theta = self.data.qpos[2]
            trajectory = BaseTrajectory(current_pos, base_pos[:2], current_theta, theta_z, duration=1.0)
            self.movement_queue.append(trajectory)
            return
        
        # Move to a point 50cm from the target
        unit_direction = direction / distance
        new_target_pos = target_pos - 0.5 * unit_direction
        
        current_pos = base_pos[:2].copy()
        current_theta = self.data.qpos[2]
        trajectory = BaseTrajectory(current_pos, new_target_pos[:2], current_theta, theta_z, duration=2.5)
        self.movement_queue.append(trajectory)

    def pick_up(self, target_name):
        """Execute a 4-phase pickup sequence for a target object."""
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        target_pos = self.data.body(target_id).xpos.copy()
        target_quat = self.data.body(target_id).xquat.copy()
        
        r = Rotation.from_quat(_mujoco_to_scipy_quat(target_quat))
        target_rpy = r.as_euler('xyz', degrees=False)
        
        pickup_sequence = PickupSequence(target_name, target_pos, target_rpy)
        self.movement_queue.append(pickup_sequence)
    
    def place(self):
        """Release gripper to place the held object."""
        current_gripper = self.data.ctrl[10]
        trajectory = GripperTrajectory(current_gripper, 0, duration=1.0, description="Open gripper to place")
        self.movement_queue.append(trajectory)

    def get_cube_positions(self):
        """Get current positions and orientations of all cubes."""
        cube_list = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not (body_name and body_name.startswith('cube_')):
                continue

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = self.data.body(body_id).xpos
            quat = self.data.body(body_id).xquat

            r = Rotation.from_quat(_mujoco_to_scipy_quat(quat))
            
            cube_info = {
                'name': body_name,
                'position': np.round(pos, 2).tolist(),
                'euler': np.round(r.as_euler('xyz', degrees=False), 2).tolist()
            }
            cube_list.append(cube_info)
        return cube_list

    def set_cube_positions(self):
        """Randomize cube positions and orientations in the scene."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not (body_name and body_name.startswith('cube_')):
                continue

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            jnt_adr = self.model.body_jntadr[body_id]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]

            self.data.qpos[qpos_adr:qpos_adr+2] = np.random.uniform(-4.5, 4.5, size=2)
            
            random_z_angle = np.random.uniform(0, 2 * np.pi)
            r = Rotation.from_euler('z', random_z_angle)
            self.data.qpos[qpos_adr+3:qpos_adr+7] = _scipy_to_mujoco_quat(r.as_quat())


    def _update_trajectory_queue(self):
        """Update the trajectory queue and execute the current movement."""
        if not self.current_trajectory and self.movement_queue:
            self.current_trajectory = self.movement_queue.pop(0)
            self.last_trajectory_time = time.time()
        
        if self.current_trajectory:
            trajectory = self.current_trajectory
            update_intervals = {'arm': 0.01, 'base': 0.04, 'gripper': 0.01, 'pickup_sequence': 0.01}
            update_interval = update_intervals.get(trajectory.type, 0.01)
            
            if time.time() - self.last_trajectory_time >= update_interval:
                self._execute_trajectory_step()
                self.last_trajectory_time = time.time()

    def _execute_trajectory_step(self):
        """Execute one step of the current trajectory."""
        if not self.current_trajectory:
            return
        
        trajectory = self.current_trajectory
        
        if trajectory.type == 'pickup_sequence':
            self._execute_pickup_sequence_step(trajectory)
            return
        
        if trajectory.current_step >= trajectory.steps:
            self.current_trajectory = None
            return
        
        alpha = trajectory.current_step / (trajectory.steps - 1) if trajectory.steps > 1 else 1.0
        smooth_alpha = _s_curve_interpolation(alpha)
        
        if trajectory.type == 'arm':
            # Use IK to get target joints from target position and orientation
            target_joints = self.util.solve_ik(trajectory.target_pos, trajectory.target_rpy, self.data)
            current_joints = self.data.qpos[3:10].copy()
            interpolated = current_joints + smooth_alpha * (target_joints - current_joints)
            self.data.ctrl[3:10] = interpolated
        elif trajectory.type == 'base':
            interpolated_pos = trajectory.start_pos + smooth_alpha * (trajectory.target_pos - trajectory.start_pos)
            interpolated_theta = trajectory.start_theta + smooth_alpha * (trajectory.target_theta - trajectory.start_theta)
            self.data.ctrl[0:2] = interpolated_pos
            self.data.ctrl[2] = interpolated_theta
        elif trajectory.type == 'gripper':
            interpolated_gripper = trajectory.start_gripper + smooth_alpha * (trajectory.target_gripper - trajectory.start_gripper)
            self.data.ctrl[10] = interpolated_gripper
        
        trajectory.current_step += 1

    def _execute_pickup_sequence_step(self, sequence):
        """Execute one step of a pickup sequence."""
        if sequence.current_trajectory_index >= len(sequence.trajectories):
            self.current_trajectory = None
            return
        
        current_trajectory = sequence.trajectories[sequence.current_trajectory_index]
        
        # Check if current sub-trajectory is finished
        if current_trajectory.current_step >= current_trajectory.steps:
            sequence.current_trajectory_index += 1
            return
        
        # Execute the current sub-trajectory step
        alpha = current_trajectory.current_step / (current_trajectory.steps - 1) if current_trajectory.steps > 1 else 1.0
        smooth_alpha = _s_curve_interpolation(alpha)
        
        if current_trajectory.type == 'arm':
            target_joints = self.util.solve_ik(current_trajectory.target_pos, current_trajectory.target_rpy, self.data)
            current_joints = self.data.qpos[3:10].copy()
            interpolated = current_joints + smooth_alpha * (target_joints - current_joints)
            self.data.ctrl[3:10] = interpolated
        elif current_trajectory.type == 'gripper':
            interpolated_gripper = current_trajectory.start_gripper + smooth_alpha * (current_trajectory.target_gripper - current_trajectory.start_gripper)
            self.data.ctrl[10] = interpolated_gripper
        
        current_trajectory.current_step += 1

def main():
    """Main function for standalone execution."""
    model_path = "../model/stanford_tidybot/tidybot.xml"
    simulator = MujocoSimulator(model_path)
    simulator.set_cube_positions()
    simulator.run()


if __name__ == "__main__":
    main()
