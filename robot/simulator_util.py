import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


class IKSolver:
    """Utility class for MuJoCo simulation with inverse kinematics solver."""
    
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