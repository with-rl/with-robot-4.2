"""
Mobile robot functionality for the MuJoCo TidyBot simulation.
Handles mobile base movement and trajectory execution.
"""

import time
import numpy as np
import mujoco
from manipulator import BaseTrajectory


# --- Mobile Robot Class ---

class MobileRobot:
    """Mobile robot class handling base movement and trajectory execution."""
    
    def __init__(self, model, data):
        """Initialize the mobile robot with MuJoCo model and data."""
        self.model = model
        self.data = data
        self.movement_queue = []
        self.current_trajectory = None
        self.last_trajectory_time = 0.0
    
    # --- Public Methods: Mobile Base Actions ---

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
    
    # --- Private Methods: Trajectory Management ---

    def update_trajectory_queue(self):
        """Update the trajectory queue and execute the current movement."""
        if not self.current_trajectory and self.movement_queue:
            self.current_trajectory = self.movement_queue.pop(0)
            self.last_trajectory_time = time.time()
        
        if self.current_trajectory:
            trajectory = self.current_trajectory
            # Base trajectories use 0.04 second intervals
            update_interval = 0.04 if trajectory.type == 'base' else 0.01
            
            if time.time() - self.last_trajectory_time >= update_interval:
                self.execute_trajectory_step()
                self.last_trajectory_time = time.time()
    
    def execute_trajectory_step(self):
        """Execute one step of the current trajectory."""
        if not self.current_trajectory:
            return
        
        trajectory = self.current_trajectory
        
        if trajectory.type == 'base':
            # Execute base trajectory using step-based interpolation like the original
            if trajectory.current_step >= trajectory.steps:
                # Trajectory completed
                self.current_trajectory = None
                return
            
            # Use S-curve interpolation like the original TrajectoryExecutor
            alpha = trajectory.current_step / (trajectory.steps - 1) if trajectory.steps > 1 else 1.0
            smooth_alpha = self._s_curve_interpolation(alpha)
            
            # Interpolate position and orientation
            interpolated_pos = trajectory.start_pos + smooth_alpha * (trajectory.target_pos - trajectory.start_pos)
            
            # Handle angle interpolation (shortest path)
            angle_diff = trajectory.target_theta - trajectory.start_theta
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            interpolated_theta = trajectory.start_theta + smooth_alpha * angle_diff
            
            # Use ctrl instead of qpos for proper control
            self.data.ctrl[0:2] = interpolated_pos
            self.data.ctrl[2] = interpolated_theta
            
            trajectory.current_step += 1
    
    def _s_curve_interpolation(self, alpha):
        """Calculate S-curve (smoothstep) interpolation for smooth trajectories."""
        return 3 * alpha**2 - 2 * alpha**3