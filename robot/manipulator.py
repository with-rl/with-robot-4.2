"""
Robot manipulator functionality for the MuJoCo TidyBot simulation.
Handles inverse kinematics, trajectory execution, and arm control.
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


# --- Helper Functions ---


def _s_curve_interpolation(alpha):
    """Calculate S-curve (smoothstep) interpolation for smooth trajectories."""
    return 3 * alpha**2 - 2 * alpha**3


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

    def __init__(
        self,
        start_pos,
        target_pos,
        start_theta,
        target_theta,
        duration,
        description="Base movement",
    ):
        steps = int(duration * 25)  # 25Hz
        super().__init__("base", steps, description)
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.start_theta = start_theta
        self.target_theta = target_theta
        self.duration = duration


class GripperTrajectory(Trajectory):
    """Trajectory for gripper movements (open/close)."""

    def __init__(
        self, start_gripper, target_gripper, duration, description="Gripper movement"
    ):
        steps = int(duration * 100)  # 100Hz
        super().__init__("gripper", steps, description)
        self.start_gripper = start_gripper
        self.target_gripper = target_gripper


class ArmTrajectory(Trajectory):
    """Trajectory for robot arm movements using inverse kinematics."""

    def __init__(self, target_pos, target_rpy, duration, description="Arm movement"):
        steps = int(duration * 100)  # 100Hz
        super().__init__("arm", steps, description)
        self.target_pos = target_pos
        self.target_rpy = target_rpy


class PickupSequence(Trajectory):
    """
    로봇 팔의 다단계 잡기 동작을 관리합니다.
    사용자가 요청한 5단계 잡기 프로세스를 따릅니다.
    """

    # 잡기 시퀀스 상수
    APPROACH_HEIGHT_OFFSET = 0.2  # 접근 시 목표물 위로의 높이 (m)
    GRASP_HEIGHT_OFFSET = 0.025  # 잡기 시 목표물 위로의 높이 (m)
    LIFT_HEIGHT_OFFSET = 0.2  # 들어올릴 높이 (m)

    GRIPPER_OPEN_DURATION = 0.5  # 그리퍼 열기 시간 (s)
    APPROACH_DURATION = 2.0  # 접근 시간 (s)
    FINAL_MOVE_DURATION = 2.5  # 최종 이동 시간 (s)
    ADJUSTMENT_DURATION = 1.5  # 위치 조정 시간 (s)
    LIFT_DURATION = 2.0  # 들어올리기 시간 (s)
    GRIPPER_CLOSE_DURATION = 1.0  # 그리퍼 닫기 시간 (s)

    def __init__(self, target, initial_target_pos, target_rpy):
        super().__init__("pickup_sequence", 999999, f"{target} 잡기 시퀀스")
        self.target_name = target
        self.initial_target_pos = initial_target_pos
        self.target_rpy = target_rpy
        self.phase = "START"  # 시퀀스 시작 단계
        self.last_grasp_pos = None  # 보정된 잡기 위치 저장

        # 첫 번째 동작: 그리퍼 열기
        self.trajectories = [
            GripperTrajectory(255, 0, self.GRIPPER_OPEN_DURATION, "그리퍼 열기")
        ]
        self.current_trajectory_index = 0

    def get_next_trajectory(self, model, data):
        """현재 동작 완료 시 다음 동작을 생성하여 반환합니다."""
        if self.phase == "START":
            # 1단계: 목표물의 예상 위치 위쪽으로 이동
            self.phase = "STEP_1_APPROACH"
            approach_pos = self.initial_target_pos + np.array(
                [0, 0, self.APPROACH_HEIGHT_OFFSET]
            )
            return ArmTrajectory(
                approach_pos,
                self.target_rpy,
                self.APPROACH_DURATION,
                f"1단계: {self.target_name} 접근",
            )
            
        elif self.phase == "STEP_1_APPROACH":
            # 2단계: 목표물의 현재 위치 재측정
            target_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.target_name
            )
            current_target_pos = data.body(target_id).xpos.copy()
            print(f"  [2단계-보정 1] 목표물 위치 재측정: {current_target_pos[:2]}")

            # 3단계: 재측정된 위치를 기반으로 잡기 직전까지 이동
            self.phase = "STEP_3_FINAL_MOVE"
            grasp_pos = current_target_pos + np.array([0, 0, self.GRASP_HEIGHT_OFFSET])
            self.last_grasp_pos = grasp_pos  # 보정된 위치 저장

            return ArmTrajectory(
                grasp_pos,
                self.target_rpy,
                self.FINAL_MOVE_DURATION,
                f"3단계: {self.target_name} 잡기 위치로 이동",
            )
            
        elif self.phase == "STEP_3_FINAL_MOVE":
            # 4단계: 위치 한 번 더 조정
            self.phase = "STEP_4_ADJUST"
            target_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.target_name
            )
            current_target_pos = data.body(target_id).xpos.copy()
            print(f"  [4단계-보정 2] 목표물 위치 최종 조정: {current_target_pos[:2]}")

            final_grasp_pos = current_target_pos + np.array(
                [0, 0, self.GRASP_HEIGHT_OFFSET]
            )
            self.last_grasp_pos = final_grasp_pos  # 가장 최근 위치로 업데이트

            return ArmTrajectory(
                final_grasp_pos,
                self.target_rpy,
                self.ADJUSTMENT_DURATION,
                f"4단계: {self.target_name} 위치 최종 조정",
            )

        elif self.phase == "STEP_4_ADJUST":
            # 잡기 동작 (그리퍼 닫기)
            self.phase = "GRIP"
            return GripperTrajectory(
                0, 250, self.GRIPPER_CLOSE_DURATION, "그리퍼 닫기"
            )

        elif self.phase == "GRIP":
            # 5단계: 목표물 들어올리기
            self.phase = "STEP_5_LIFT"
            if self.last_grasp_pos is None:
                print("오류: 마지막 잡기 위치가 설정되지 않아 들어올릴 수 없습니다.")
                return None

            lift_pos = self.last_grasp_pos + np.array(
                [0, 0, self.LIFT_HEIGHT_OFFSET - self.GRASP_HEIGHT_OFFSET]
            )
            return ArmTrajectory(
                lift_pos, self.target_rpy, self.LIFT_DURATION, f"5단계: {self.target_name} 들어올리기"
            )

        elif self.phase == "STEP_5_LIFT":
            # 시퀀스 완료
            self.phase = "DONE"
            return None

        return None


# --- Inverse Kinematics Solver ---


class IKSolver:
    """Inverse kinematics solver for robot arm control and positioning."""

    # Weights for the IK cost function
    POSITION_ERROR_WEIGHT = 5.0
    ROTATION_ERROR_WEIGHT = 0.01

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_names = [f"joint_{i}" for i in range(1, 8)]
        self.joint_bounds = self._get_joint_bounds()

    def _get_joint_bounds(self):
        """Get joint limits from the MuJoCo model."""
        bounds = []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            jnt_range = self.model.jnt_range[joint_id]
            # If range is not defined (0, 0), assume unlimited
            if jnt_range[0] == 0 and jnt_range[1] == 0:
                bounds.append((-np.inf, np.inf))
            else:
                bounds.append(tuple(jnt_range))
        return bounds

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
        rotation_error = np.linalg.norm(current_mat - target_mat, ord="fro")
        return (
            self.POSITION_ERROR_WEIGHT * position_error
            + self.ROTATION_ERROR_WEIGHT * rotation_error
        )

    def solve_ik(self, target_pos, target_rpy, real_data):
        """
        Solve inverse kinematics for target position and orientation.
        """
        # Initialize with current robot state
        self.data.qpos[:] = real_data.qpos[:]
        initial_joints = np.copy(self.data.qpos[3:10])

        target_rpy[0] = np.pi
        target_rpy[1] = 0
        target_mat = R.from_euler("xyz", target_rpy).as_matrix()

        # Solve optimization
        result = minimize(
            self.ik_cost,
            initial_joints,
            args=(target_pos, target_mat),
            bounds=self.joint_bounds,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 1000},
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

        if trajectory.type == "pickup_sequence":
            return self._execute_pickup_sequence_step(trajectory, data)

        if trajectory.current_step >= trajectory.steps:
            return True  # Trajectory completed

        alpha = (
            trajectory.current_step / (trajectory.steps - 1)
            if trajectory.steps > 1
            else 1.0
        )
        smooth_alpha = _s_curve_interpolation(alpha)

        if trajectory.type == "arm":
            # Use IK to get target joints from target position and orientation
            target_joints = self.ik_solver.solve_ik(
                trajectory.target_pos, trajectory.target_rpy, data
            )
            current_joints = data.qpos[3:10].copy()
            interpolated = current_joints + smooth_alpha * (
                target_joints - current_joints
            )
            data.ctrl[3:10] = interpolated
        elif trajectory.type == "base":
            interpolated_pos = trajectory.start_pos + smooth_alpha * (
                trajectory.target_pos - trajectory.start_pos
            )
            interpolated_theta = trajectory.start_theta + smooth_alpha * (
                trajectory.target_theta - trajectory.start_theta
            )
            data.ctrl[0:2] = interpolated_pos
            data.ctrl[2] = interpolated_theta
        elif trajectory.type == "gripper":
            interpolated_gripper = trajectory.start_gripper + smooth_alpha * (
                trajectory.target_gripper - trajectory.start_gripper
            )
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
            # Print phase completion details
            current_pos = data.site("pinch_site").xpos.copy()
            print(f"[PHASE COMPLETED] {current_trajectory.description}", flush=True)
            print(
                f"  Final End-Effector Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]",
                flush=True,
            )

            # If it's an arm trajectory, calculate and print the position error
            if current_trajectory.type == "arm":
                target_pos = current_trajectory.target_pos
                position_error = target_pos - current_pos
                error_distance = np.linalg.norm(position_error)
                print(
                    f"  Target Position:              [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]",
                    flush=True,
                )
                print(
                    f"  Position Error (XYZ):         [{position_error[0]:.4f}, {position_error[1]:.4f}, {position_error[2]:.4f}]m",
                    flush=True,
                )
                print(
                    f"  Error Distance:               {error_distance * 1000:.2f}mm",
                    flush=True,
                )

            print("  " + "=" * 50, flush=True)

            # Get the next trajectory dynamically
            next_trajectory = sequence.get_next_trajectory(self.ik_solver.model, data)
            if next_trajectory:
                sequence.trajectories.append(next_trajectory)
                sequence.current_trajectory_index += 1
            else:
                # End of sequence
                sequence.current_trajectory_index += 1  # To make it out of bounds
                return True

            return False  # Continue sequence

        # Execute the current sub-trajectory step
        alpha = (
            current_trajectory.current_step / (current_trajectory.steps - 1)
            if current_trajectory.steps > 1
            else 1.0
        )
        smooth_alpha = _s_curve_interpolation(alpha)

        if current_trajectory.type == "arm":
            target_joints = self.ik_solver.solve_ik(
                current_trajectory.target_pos, current_trajectory.target_rpy, data
            )
            current_joints = data.qpos[3:10].copy()
            interpolated = current_joints + smooth_alpha * (
                target_joints - current_joints
            )
            data.ctrl[3:10] = interpolated
        elif current_trajectory.type == "gripper":
            interpolated_gripper = current_trajectory.start_gripper + smooth_alpha * (
                current_trajectory.target_gripper - current_trajectory.start_gripper
            )
            data.ctrl[10] = interpolated_gripper

        current_trajectory.current_step += 1
        return False  # Sequence still running
