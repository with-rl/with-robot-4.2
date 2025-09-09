"""
Robot manipulator functionality for the MuJoCo TidyBot simulation.
Handles inverse kinematics, trajectory execution, and arm control.
"""

import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


# --- Helper Functions ---

def _s_curve_interpolation(alpha):
    """Calculate S-curve (smoothstep) interpolation for smooth trajectories."""
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
        self.duration = duration


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


# --- Inverse Kinematics Solver ---

class IKSolver:
    """Inverse kinematics solver for robot arm control and positioning."""
    
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
    
    def fk(self, joints):
        """Forward kinematics: compute end-effector pose from joint angles."""
        self.data.qpos[3:10] = joints
        mujoco.mj_kinematics(self.model, self.data)
        
        pos = self.data.site("pinch_site").xpos
        mat = self.data.site("pinch_site").xmat.reshape(3, 3)
        return pos, mat

    def ik_cost(self, thetas, target_pos, target_mat):
        """Cost function for IK optimization."""
        current_pos, current_mat = self.fk(thetas)
        position_error = np.linalg.norm(target_pos - current_pos)
        rotation_error = 0.5 * np.linalg.norm(current_mat - target_mat, ord='fro')
        return position_error + rotation_error

    def solve_ik(self, target_pos, target_rpy, real_data):
        """
        Solve inverse kinematics for target position and orientation.
        """
        # Initialize with current robot state
        self.data.qpos[:] = real_data.qpos[:]
        initial_joints = np.copy(self.data.qpos[3:10])
        
        # Joint limits from TidyBot XML model
        joint_bounds = [
            (-np.inf, np.inf),    # joint_1: unlimited
            (-2.24, 2.24),        # joint_2: ±128.4°
            (-np.inf, np.inf),    # joint_3: unlimited  
            (-2.57, 2.57),        # joint_4: ±147.3°
            (-np.inf, np.inf),    # joint_5: unlimited
            (-2.09, 2.09),        # joint_6: ±119.7°
            (-np.inf, np.inf),    # joint_7: unlimited
        ]
        
        target_rpy[0] = np.pi
        target_rpy[1] = 0
        target_mat = R.from_euler('xyz', target_rpy).as_matrix()
        
        # Solve optimization
        result = minimize(
            self.ik_cost,
            initial_joints,
            args=(target_pos, target_mat),
            bounds=joint_bounds,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 1000}
        )
        
        # Validate solution
        achieved_pos, _ = self.fk(result.x)
        position_error = np.linalg.norm(target_pos - achieved_pos)
        
        if position_error > 0.02:  # 2cm threshold
            print(f"Warning: IK position error {position_error:.3f}m")
        
        return result.x


# --- Trajectory Executor ---

class TrajectoryExecutor:
    """Trajectory executor for robot arm and gripper movements."""
    
    def __init__(self, ik_solver):
        """Initialize the trajectory executor with an IK solver."""
        self.ik_solver = ik_solver
    
    def execute_trajectory_step(self, trajectory, data):
        """Execute one step of a trajectory."""
        if not trajectory:
            return False
        
        if trajectory.type == 'pickup_sequence':
            return self._execute_pickup_sequence_step(trajectory, data)
        
        if trajectory.current_step >= trajectory.steps:
            return True  # Trajectory completed
        
        alpha = trajectory.current_step / (trajectory.steps - 1) if trajectory.steps > 1 else 1.0
        smooth_alpha = _s_curve_interpolation(alpha)
        
        if trajectory.type == 'arm':
            # Use IK to get target joints from target position and orientation
            target_joints = self.ik_solver.solve_ik(trajectory.target_pos, trajectory.target_rpy, data)
            current_joints = data.qpos[3:10].copy()
            interpolated = current_joints + smooth_alpha * (target_joints - current_joints)
            data.ctrl[3:10] = interpolated
        elif trajectory.type == 'base':
            interpolated_pos = trajectory.start_pos + smooth_alpha * (trajectory.target_pos - trajectory.start_pos)
            interpolated_theta = trajectory.start_theta + smooth_alpha * (trajectory.target_theta - trajectory.start_theta)
            data.ctrl[0:2] = interpolated_pos
            data.ctrl[2] = interpolated_theta
        elif trajectory.type == 'gripper':
            interpolated_gripper = trajectory.start_gripper + smooth_alpha * (trajectory.target_gripper - trajectory.start_gripper)
            data.ctrl[10] = interpolated_gripper
        
        trajectory.current_step += 1
        return False  # Trajectory still running

    def _execute_pickup_sequence_step(self, sequence, data):
        """Execute one step of a pickup sequence."""
        if sequence.current_trajectory_index >= len(sequence.trajectories):
            return True  # Sequence completed
        
        current_trajectory = sequence.trajectories[sequence.current_trajectory_index]
        
        # Check if current sub-trajectory is finished
        if current_trajectory.current_step >= current_trajectory.steps:
            sequence.current_trajectory_index += 1
            return False
        
        # Execute the current sub-trajectory step
        alpha = current_trajectory.current_step / (current_trajectory.steps - 1) if current_trajectory.steps > 1 else 1.0
        smooth_alpha = _s_curve_interpolation(alpha)
        
        if current_trajectory.type == 'arm':
            target_joints = self.ik_solver.solve_ik(current_trajectory.target_pos, current_trajectory.target_rpy, data)
            current_joints = data.qpos[3:10].copy()
            interpolated = current_joints + smooth_alpha * (target_joints - current_joints)
            data.ctrl[3:10] = interpolated
        elif current_trajectory.type == 'gripper':
            interpolated_gripper = current_trajectory.start_gripper + smooth_alpha * (current_trajectory.target_gripper - current_trajectory.start_gripper)
            data.ctrl[10] = interpolated_gripper
        
        current_trajectory.current_step += 1
        return False  # Sequence still running