import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


class MujocoSimulatorUtil:
    def __init__(self, model):
        """
        Initializes the simulator util.
        """
        self.model = model
        self.data = mujoco.MjData(model)  # MuJoCo data
    
    def fk(self, joints):
        self.data.qpos[3:10] = joints
        mujoco.mj_kinematics(self.model, self.data)

        pos = self.data.site("pinch_site").xpos
        mat = self.data.site("pinch_site").xmat.reshape(3, 3)
        return pos, mat

    def ik_cost(self, thetas, pos, mat):
        pos_hat, mat_hat = self.fk(thetas)
        p_error = np.linalg.norm(pos - pos_hat)
        r_error = 0.5 * np.linalg.norm(mat_hat - mat, ord='fro')
        return p_error + r_error

    def solve_ik(self, pos, rpy, real_data):
        self.data.qpos[:] = real_data.qpos[:]

        initial_thetas = np.copy(self.data.qpos[3:10])
        theta_bounds = [
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
        ]
        mat = R.from_euler('xyz', rpy).as_matrix()

        result = minimize(
            self.ik_cost,  # 목적 함수
            initial_thetas,  # 초기값
            args=(pos, mat),  # 추가 매개변수
            bounds=theta_bounds,  # 범위 제한
            method="SLSQP",  # 제약 조건을 지원하는 최적화 알고리즘
            options={"ftol": 1e-6, "maxiter": 500},  # 수렴 기준
        )
        return result.x