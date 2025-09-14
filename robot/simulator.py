"""
Class-based code for the MuJoCo TidyBot simulation.
This should be run with mjpython: mjpython simulator.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
from manipulator import (
    IKSolver,
    TrajectoryExecutor,
    GripperTrajectory,
    PickupSequence
)
from mobile import MobileRobot
from scipy.spatial.transform import Rotation


# --- Helper Functions ---


def _scipy_to_mujoco_quat(scipy_quat):
    """Convert SciPy quaternion [x, y, z, w] to MuJoCo quaternion [w, x, y, z]."""
    return np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])


def _mujoco_to_scipy_quat(mujoco_quat):
    """Convert MuJoCo quaternion [w, x, y, z] to SciPy quaternion [x, y, z, w]."""
    return np.array([mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]])


# --- Main Simulator Class ---


class MujocoSimulator:
    """Main simulator class handling robot control and movement trajectories."""

    # Simulation parameters
    UPDATE_INTERVALS = {"arm": 0.01, "gripper": 0.01, "pickup_sequence": 0.01}
    DEFAULT_UPDATE_INTERVAL = 0.01
    PLACE_DURATION = 1.0

    # Environment parameters
    FLOOR_BOUNDS = (-4.5, 4.5)

    def __init__(self, model_path):
        """Initialize the simulator with robot model."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.ik = IKSolver(self.model)
        self.trajectory_executor = TrajectoryExecutor(self.ik)
        self.mobile_robot = MobileRobot(self.model, self.data)

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
        occupancy_map = self.get_map()
        self.mobile_robot.move_to(target_name, occupancy_map)

    def pick_up(self, target_name):
        """Execute a 4-phase pickup sequence for a target object."""
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        target_pos = self.data.body(target_id).xpos.copy()
        target_quat = self.data.body(target_id).xquat.copy()

        r = Rotation.from_quat(_mujoco_to_scipy_quat(target_quat))
        target_rpy = r.as_euler("xyz", degrees=False)

        pickup_sequence = PickupSequence(target_name, target_pos, target_rpy)
        self.movement_queue.append(pickup_sequence)

    def place(self):
        """Release gripper to place the held object."""
        current_gripper = self.data.ctrl[10]
        trajectory = GripperTrajectory(
            current_gripper, 0, duration=self.PLACE_DURATION, description="Open gripper to place"
        )
        self.movement_queue.append(trajectory)

    def get_map(self):
        """Generate a 100x100 occupancy grid map for 10m x 10m area.
        Each cell represents 0.1m x 0.1m. Value 1 indicates occupied space.
        Map coordinates: (-5,-5) to (5,5) in world coordinates.

        Inflates obstacles by the robot's radius for path planning.
        """
        occupancy_map = np.zeros((100, 100))

        # Inflate obstacles by robot radius (approx 0.27m) for C-space planning
        robot_safety_margin = 0.0

        # Helper function to mark rectangular areas as occupied with robot safety margin
        def mark_rectangle(center_x, center_y, size_x, size_y):
            # Expand obstacle bounds by robot safety margin
            expanded_size_x = size_x + robot_safety_margin
            expanded_size_y = size_y + robot_safety_margin

            # Calculate grid bounds
            x_min = max(0, int((center_x - expanded_size_x + 5.0) * 10))
            x_max = min(100, int((center_x + expanded_size_x + 5.0) * 10) + 1)
            y_min = max(0, int((center_y - expanded_size_y + 5.0) * 10))
            y_max = min(100, int((center_y + expanded_size_y + 5.0) * 10) + 1)

            occupancy_map[x_min:x_max, y_min:y_max] = 1

        # Mark all movable objects using their actual geom sizes + robot safety margin
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            # Only world and _ (object)
            if not (
                body_name
                and (body_name.startswith("world") or body_name.startswith("_"))
            ):
                continue

            # Find geoms associated with this body and get their sizes
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == body_id:
                    geom_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    if geom_name in ["floor"]:
                        continue
                    # geom_type = self.model.geom_type[geom_id]
                    geom_pos = self.data.geom_xpos[geom_id]
                    x, y = geom_pos[0], geom_pos[1]
                    geom_size = self.model.geom_size[geom_id]
                    size_x, size_y = geom_size[0], geom_size[1]
                    mark_rectangle(x, y, size_x, size_y)

        return occupancy_map

    def wait_for_completion(self):
        """Wait for all queued movements to complete."""
        import time

        while (
            self.movement_queue
            or self.current_trajectory
            or self.mobile_robot.movement_queue
            or self.mobile_robot.current_trajectory
        ):
            time.sleep(0.1)

    def is_busy(self):
        """Check if robot is currently executing any movements."""
        return (
            bool(self.movement_queue)
            or bool(self.current_trajectory)
            or bool(self.mobile_robot.movement_queue)
            or bool(self.mobile_robot.current_trajectory)
        )

    def get_cube_positions(self):
        """Get current positions and orientations of all cubes."""
        cube_list = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not (body_name and body_name.startswith("_cube_")):
                continue

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = self.data.body(body_id).xpos
            quat = self.data.body(body_id).xquat

            r = Rotation.from_quat(_mujoco_to_scipy_quat(quat))

            cube_info = {
                "name": body_name,
                "position": np.round(pos, 2).tolist(),
                "euler": np.round(r.as_euler("xyz", degrees=False), 2).tolist(),
            }
            cube_list.append(cube_info)
        return cube_list

    def set_cube_positions(self):
        """Randomize cube positions and orientations in the scene."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not (body_name and body_name.startswith("_cube_")):
                continue

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            jnt_adr = self.model.body_jntadr[body_id]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]

            self.data.qpos[qpos_adr : qpos_adr + 2] = np.random.uniform(
                self.FLOOR_BOUNDS[0], self.FLOOR_BOUNDS[1], size=2
            )

            random_z_angle = np.random.uniform(0, 2 * np.pi)
            r = Rotation.from_euler("z", random_z_angle)
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = _scipy_to_mujoco_quat(
                r.as_quat()
            )

    def _update_trajectory_queue(self):
        """Update the trajectory queue and execute the current movement."""
        # Update mobile robot trajectory queue first
        self.mobile_robot.update_trajectory_queue()

        # Handle non-mobile trajectories (arm, gripper, pickup sequences)
        if not self.current_trajectory and self.movement_queue:
            self.current_trajectory = self.movement_queue.pop(0)
            self.last_trajectory_time = time.time()

        if self.current_trajectory:
            trajectory = self.current_trajectory
            update_interval = self.UPDATE_INTERVALS.get(
                trajectory.type, self.DEFAULT_UPDATE_INTERVAL
            )

            if time.time() - self.last_trajectory_time >= update_interval:
                self._execute_trajectory_step()
                self.last_trajectory_time = time.time()

    def _execute_trajectory_step(self):
        """Execute one step of the current trajectory."""
        if not self.current_trajectory:
            return

        # Use the TrajectoryExecutor to handle trajectory execution
        trajectory_completed = self.trajectory_executor.execute_trajectory_step(
            self.current_trajectory, self.data
        )

        if trajectory_completed:
            self.current_trajectory = None


def main():
    """Main function for standalone execution."""
    model_path = "../model/stanford_tidybot/tidybot.xml"
    simulator = MujocoSimulator(model_path)
    simulator.set_cube_positions()
    simulator.run()


if __name__ == "__main__":
    main()
