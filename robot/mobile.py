"""
Mobile robot functionality for the MuJoCo TidyBot simulation.
Handles mobile base movement and trajectory execution.
"""

import heapq
import time

import mujoco
import numpy as np
from manipulator import BaseTrajectory
from PIL import Image, ImageDraw


# --- A* Pathfinding Algorithm ---


class AStarPathfinder:
    """A* pathfinding algorithm for grid-based navigation."""

    # Cost for movement
    STRAIGHT_COST = 1.0
    DIAGONAL_COST = 1.414

    # Grid conversion parameters
    WORLD_OFFSET = 5.0  # Half the world size
    GRID_SCALE = 10  # Cells per meter

    def __init__(self, occupancy_map):
        """Initialize pathfinder with occupancy grid map."""
        self.map = occupancy_map
        self.height, self.width = occupancy_map.shape
        self.robot_radius_cells = 3

    def is_area_safe(self, grid_x, grid_y):
        """Checks if the area for the robot's footprint around a grid cell is free of obstacles."""
        for rdx in range(-self.robot_radius_cells, self.robot_radius_cells + 1):
            for rdy in range(-self.robot_radius_cells, self.robot_radius_cells + 1):
                check_x, check_y = grid_x + rdx, grid_y + rdy
                if not (
                    0 <= check_x < self.width
                    and 0 <= check_y < self.height
                    and self.map[check_x, check_y] == 0
                ):
                    return False
        return True

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        grid_x = int((x + self.WORLD_OFFSET) * self.GRID_SCALE)
        grid_y = int((y + self.WORLD_OFFSET) * self.GRID_SCALE)
        return np.clip(grid_x, 0, self.width - 1), np.clip(grid_y, 0, self.height - 1)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates."""
        x = (grid_x / self.GRID_SCALE) - self.WORLD_OFFSET
        y = (grid_y / self.GRID_SCALE) - self.WORLD_OFFSET
        return x, y

    def get_neighbors(self, x, y):
        """Get valid neighboring cells (8-connected)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_area_safe(nx, ny):
                    cost = (
                        self.DIAGONAL_COST if dx != 0 and dy != 0 else self.STRAIGHT_COST
                    )
                    neighbors.append((nx, ny, cost))
        return neighbors

    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def find_path(self, start_world, goal_world):
        """Find optimal path from start to goal using A* algorithm."""
        start_grid = self.world_to_grid(start_world[0], start_world[1])
        goal_grid = self.world_to_grid(goal_world[0], goal_world[1])

        if not self.is_area_safe(goal_grid[0], goal_grid[1]):
            return None

        # A* algorithm
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    x, y = self.grid_to_world(current[0], current[1])
                    path.append([x, y])
                    current = came_from[current]
                # Add start position
                x, y = self.grid_to_world(start_grid[0], start_grid[1])
                path.append([x, y])
                path.reverse()
                return path

            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(
                current[0], current[1]
            ):
                neighbor = (neighbor_x, neighbor_y)
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal_grid
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found


# --- Mobile Robot Class ---


class MobileRobot:
    """Mobile robot class handling base movement and trajectory execution."""

    # Path planning parameters
    APPROACH_DISTANCE = 0.65  # m
    FALLBACK_DISTANCE_MULTIPLIER = 0.5
    GOAL_SEARCH_ANGLE_STEP = 15  # degrees

    # Trajectory parameters
    ROBOT_BASE_SPEED = 5.0  # m/s
    MIN_TRAJECTORY_DURATION = 0.2  # seconds
    GOAL_DISTANCE_THRESHOLD = 0.05  # m
    MIN_ROTATION_ANGLE = 0.1  # radians
    ROTATION_DURATION = 1.0  # seconds

    # Simulation parameters
    BASE_UPDATE_INTERVAL = 0.04  # seconds
    DEFAULT_UPDATE_INTERVAL = 0.01  # seconds

    def __init__(self, model, data):
        """Initialize the mobile robot with MuJoCo model and data."""
        self.model = model
        self.data = data
        self.movement_queue = []
        self.current_trajectory = None
        self.last_trajectory_time = 0.0

    # --- Public Methods: Mobile Base Actions ---

    def move_to(self, target_name, occupancy_map, display_map=False):
        """Move robot base toward a target object using A* pathfinding, stopping near the target."""
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        target_pos = self.data.body(target_id).xpos.copy()[:2]

        base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "joint_0_site"
        )
        base_pos = self.data.site(base_id).xpos.copy()[:2]
        base_theta = self.data.qpos[2]
        print("base_pos:", base_pos)
        print("target_pos:", target_pos)

        distance_to_target = np.linalg.norm(target_pos - base_pos)
        direction = target_pos - base_pos
        theta_z = np.arctan2(direction[1], direction[0])

        if distance_to_target < self.APPROACH_DISTANCE:
            trajectory = BaseTrajectory(
                base_pos, base_pos, base_theta, theta_z, duration=self.ROTATION_DURATION
            )
            self.movement_queue.append(trajectory)
            return

        pathfinder = AStarPathfinder(occupancy_map)

        start_grid = pathfinder.world_to_grid(base_pos[0], base_pos[1])
        if occupancy_map[start_grid] == 1:
            return

        path = None
        goal_pos = None
        best_path = None
        best_distance = float("inf")

        for angle_deg in np.arange(0, 360, self.GOAL_SEARCH_ANGLE_STEP):
            angle_rad = np.deg2rad(angle_deg)
            test_goal_pos = target_pos[:2] + self.APPROACH_DISTANCE * np.array(
                [np.cos(angle_rad), np.sin(angle_rad)]
            )

            goal_grid = pathfinder.world_to_grid(test_goal_pos[0], test_goal_pos[1])
            if occupancy_map[goal_grid] == 0:
                test_path = pathfinder.find_path(base_pos, test_goal_pos)
                if test_path is not None:
                    distance_to_base = np.linalg.norm(test_goal_pos - base_pos[:2])
                    if distance_to_base < best_distance:
                        best_distance = distance_to_base
                        goal_pos = test_goal_pos
                        best_path = test_path

        path = best_path

        if path is None:
            print(f"No path found to {target_name}")
        else:
            print(f"Path to {target_name}: {len(path)} waypoints")

        if path is None:
            direction = target_pos - base_pos
            fallback_goal = target_pos[:2] - self.FALLBACK_DISTANCE_MULTIPLIER * (
                direction[:2] / np.linalg.norm(direction[:2])
            )
            fallback_distance = np.linalg.norm(fallback_goal - base_pos)
            fallback_duration = max(
                self.MIN_TRAJECTORY_DURATION, fallback_distance / self.ROBOT_BASE_SPEED
            )
            trajectory = BaseTrajectory(
                base_pos,
                fallback_goal,
                base_theta,
                base_theta,
                duration=fallback_duration,
            )
            self.movement_queue.append(trajectory)
            return

        if display_map:
            self._display_map(occupancy_map, pathfinder, path)

        self._create_path_trajectories(
            path, base_pos, base_theta, goal_pos, pathfinder, occupancy_map, target_pos
        )

    def _display_map(self, occupancy_map, pathfinder, path):
        transposed_map = occupancy_map.T
        image_array = (1 - transposed_map) * 255
        image_array = image_array.astype(np.uint8)

        scale = 5
        grid_color = 180  # Medium Gray
        h, w = image_array.shape
        new_h, new_w = h * scale, w * scale
        grid_image = np.full((new_h, new_w), grid_color, dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                grid_image[
                    y * scale + 1 : (y + 1) * scale, x * scale + 1 : (x + 1) * scale
                ] = image_array[y, x]

        flipped_grid_image = np.flipud(grid_image)
        img = Image.fromarray(flipped_grid_image, "L").convert("RGB")
        draw = ImageDraw.Draw(img)

        path_color = (0, 0, 255)
        arrow_length = 8
        arrow_angle = np.deg2rad(30)

        all_waypoints = path
        for i in range(len(all_waypoints) - 1):
            p1_world = all_waypoints[i]
            p2_world = all_waypoints[i + 1]

            p1_grid = pathfinder.world_to_grid(p1_world[0], p1_world[1])
            p2_grid = pathfinder.world_to_grid(p2_world[0], p2_world[1])

            p1_img_unflipped = (
                p1_grid[0] * scale + scale // 2,
                p1_grid[1] * scale + scale // 2,
            )
            p2_img_unflipped = (
                p2_grid[0] * scale + scale // 2,
                p2_grid[1] * scale + scale // 2,
            )

            p1_img = (p1_img_unflipped[0], new_h - 1 - p1_img_unflipped[1])
            p2_img = (p2_img_unflipped[0], new_h - 1 - p2_img_unflipped[1])

            draw.line([p1_img, p2_img], fill=path_color, width=2)

            vec = np.array(p1_img) - np.array(p2_img)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

                theta1 = arrow_angle
                rot1 = np.array(
                    [
                        [np.cos(theta1), -np.sin(theta1)],
                        [np.sin(theta1), np.cos(theta1)],
                    ]
                )

                theta2 = -arrow_angle
                rot2 = np.array(
                    [
                        [np.cos(theta2), -np.sin(theta2)],
                        [np.sin(theta2), np.cos(theta2)],
                    ]
                )

                leg1_vec = np.dot(rot1, vec) * arrow_length
                leg2_vec = np.dot(rot2, vec) * arrow_length

                leg1_end = (int(p2_img[0] + leg1_vec[0]), int(p2_img[1] + leg1_vec[1]))
                leg2_end = (int(p2_img[0] + leg2_vec[0]), int(p2_img[1] + leg2_vec[1]))

                draw.line([p2_img, leg1_end], fill=path_color, width=2)
                draw.line([p2_img, leg2_end], fill=path_color, width=2)
        return img

    def _create_path_trajectories(
        self,
        path,
        current_pos,
        start_theta,
        goal_pos,
        pathfinder,
        occupancy_map,
        target_pos,
    ):
        """Create a series of trajectories to follow the given path."""
        if len(path) < 2:
            return

        current_theta = start_theta
        all_waypoints = [current_pos] + path[1:]

        for i in range(len(all_waypoints) - 1):
            start_pos = np.array(all_waypoints[i])
            end_pos = np.array(all_waypoints[i + 1])

            distance = np.linalg.norm(end_pos - start_pos)
            duration = max(
                self.MIN_TRAJECTORY_DURATION, distance / self.ROBOT_BASE_SPEED
            )

            trajectory = BaseTrajectory(
                start_pos, end_pos, current_theta, current_theta, duration=duration
            )
            self.movement_queue.append(trajectory)

        final_waypoint = np.array(all_waypoints[-1])
        goal_distance = np.linalg.norm(final_waypoint - goal_pos)
        if goal_distance > self.GOAL_DISTANCE_THRESHOLD:
            duration = max(
                self.MIN_TRAJECTORY_DURATION, goal_distance / self.ROBOT_BASE_SPEED
            )
            trajectory = BaseTrajectory(
                final_waypoint,
                goal_pos,
                current_theta,
                current_theta,
                duration=duration,
            )
            self.movement_queue.append(trajectory)

        # Add final rotation to face the target
        direction = target_pos[:2] - goal_pos
        target_theta = np.arctan2(direction[1], direction[0])
        if (
            abs(target_theta - current_theta) > self.MIN_ROTATION_ANGLE
        ):  # Only rotate if significant angle difference
            rotation_trajectory = BaseTrajectory(
                goal_pos, goal_pos, current_theta, target_theta, duration=self.ROTATION_DURATION
            )
            self.movement_queue.append(rotation_trajectory)

    # --- Private Methods: Trajectory Management ---

    def update_trajectory_queue(self):
        """Update the trajectory queue and execute the current movement."""
        if not self.current_trajectory and self.movement_queue:
            self.current_trajectory = self.movement_queue.pop(0)
            self.last_trajectory_time = time.time()

        if self.current_trajectory:
            trajectory = self.current_trajectory

            update_interval = (
                self.BASE_UPDATE_INTERVAL
                if trajectory.type == "base"
                else self.DEFAULT_UPDATE_INTERVAL
            )

            if time.time() - self.last_trajectory_time >= update_interval:
                self.execute_trajectory_step()
                self.last_trajectory_time = time.time()

    def execute_trajectory_step(self):
        """Execute one step of the current trajectory."""
        if not self.current_trajectory:
            return

        trajectory = self.current_trajectory

        if trajectory.type == "base":
            if trajectory.current_step >= trajectory.steps:
                self.current_trajectory = None
                return

            alpha = (
                trajectory.current_step / (trajectory.steps - 1)
                if trajectory.steps > 1
                else 1.0
            )
            smooth_alpha = self._s_curve_interpolation(alpha)

            interpolated_pos = trajectory.start_pos + smooth_alpha * (
                trajectory.target_pos - trajectory.start_pos
            )

            angle_diff = trajectory.target_theta - trajectory.start_theta
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            interpolated_theta = trajectory.start_theta + smooth_alpha * angle_diff

            self.data.ctrl[0] = interpolated_pos[0]
            self.data.ctrl[1] = interpolated_pos[1]
            self.data.ctrl[2] = interpolated_theta

            trajectory.current_step += 1

    def _s_curve_interpolation(self, alpha):
        """Calculate S-curve (smoothstep) interpolation for smooth trajectories."""
        return 3 * alpha**2 - 2 * alpha**3
