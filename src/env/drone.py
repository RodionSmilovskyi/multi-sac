# pylint: disable=C0114, E0401, W0621
import os
import time
from typing import Any, Tuple
import gym
import math
from gym.core import ObsType, ActType
import numpy as np
import pybullet as p
import pybullet_data
from urdf_parser_py import urdf
from pybullet_utils import bullet_client as bc

from pid_controller import PIDController

MIN_VAL = 1e-4
DRONE_IMG_WIDTH = 256
DRONE_IMG_HEIGHT = 256
NUMBER_OF_CHANNELS = 3
MAX_DISTANCE = 4  # meters
MAX_ALTITUDE = 1  # meters
MIN_ALTITUDE = 0.04  # meters
START_ALTITUDE = 0.05
FRAME_NUMBER = 500
THRUST_TO_WEIGHT_RATIO = 4
DRONE_WEIGHT = 1
TILT_LITMIT = np.deg2rad(55)
G = 9.81
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def convert_range(
    x: float, x_min: float, x_max: float, y_min: float, y_max: float
) -> float:
    """Converts value from one range system to another"""
    return ((x - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min


class DroneEnv(gym.Env):
    """Class responsible for drone avionics"""

    def __init__(self, use_gui=True) -> None:
        self.plane_id = None

        self.drone_id = None

        self.target_id = None

        self.step_number = 0

        self.use_gui = use_gui

        self.metadata = {"render_fps": 30, "render_modes": ["human", "rgb_array"]}

        self._agent_location = np.array([0, 0, 0], dtype=np.int32)

        self.world_space = gym.spaces.Box(
            low=np.array([-20, -20, 0]),
            high=np.array([20, 20, MAX_ALTITUDE]),
            dtype=np.float32,
        )

        self.observation_space: ObsType = gym.spaces.Dict(
            {
                "drone_img": gym.spaces.Box(
                    low=0,
                    high=DRONE_IMG_WIDTH - 1,
                    shape=(DRONE_IMG_WIDTH, DRONE_IMG_HEIGHT, NUMBER_OF_CHANNELS),
                    dtype=np.uint8,
                ),
                "altitude": gym.spaces.Box(0, MAX_ALTITUDE, shape=(1,)),
                "roll": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                "pitch": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                "yaw": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                "distance": gym.spaces.Box(0, MAX_DISTANCE, shape=(1,)),
            }
        )

        self.action_space: ActType = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32,
        )

        self.drone_img = np.zeros(self.observation_space["drone_img"].shape)

        self.render_mode = "rgb_array"

        self.num_motors = 0
        self.mass = 0
        self.thrust_coefficient = 7e-9
        self.torque_coefficient = 5e-10
        self.motor_scaling = 15000.0
        self.max_rpm = 0
        self.min_rpm = 0

        # pylint: disable=c-extension-no-member
        self.client = bc.BulletClient(
            connection_mode=p.GUI if self.use_gui is True else p.DIRECT
        )

        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)

        self.client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        self.client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

        self.pid_roll = PIDController(Kp=1, Ki=0.05, Kd=0.15)
        self.pid_pitch = PIDController(Kp=1, Ki=0.05, Kd=0.15)

    def reset(self, initial_pos=None) -> ObsType:
        self.client.resetSimulation()

        self.client.setGravity(0, 0, -G)

        self.drone_img = np.zeros(self.observation_space["drone_img"].shape)

        self.step_number = 0

        self.plane_id = self.client.loadURDF("plane.urdf")

        random_position = self.world_space.sample()

        initial_dist = np.linalg.norm(random_position - np.array([0, 0, 0]))

        while initial_dist < 5:
            random_position = self.world_space.sample()
            initial_dist = np.linalg.norm(random_position - np.array([0, 0, 0]))

        collision_shape_id = self.client.createCollisionShape(
            shapeType=self.client.GEOM_MESH,
            fileName=f"{WORKING_DIRECTORY}/a_cube.obj",
        )
        visual_shape_id = self.client.createVisualShape(
            shapeType=self.client.GEOM_MESH,
            fileName=f"{WORKING_DIRECTORY}/a_cube.obj",
        )
        self.target_id = self.client.createMultiBody(
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=random_position.tolist(),
        )

        # self.drone_id = self.client.loadURDF(
        #     f"{WORKING_DIRECTORY}/drone.urdf",
        #     [0, 0, START_ALTITUDE],
        #     self._look_at([0, 0, START_ALTITUDE], random_position),
        # )

        self.drone_id = self.client.loadURDF(
            f"{WORKING_DIRECTORY}/drone.urdf",
            (
                initial_pos
                if initial_pos is not None
                else [MIN_VAL, MIN_VAL, START_ALTITUDE]
            ),
            # self._look_at([0, 0, START_ALTITUDE], [1, 0.0001, ]),
        )
        linear_velocity, _ = self.client.getBaseVelocity(self.drone_id)
        self.set_drone_params()

        return {
            "drone_img": self.drone_img,
            "distance": 1,
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
            "altitude": self._get_altitude(),
        }, {"vertical_velocity": linear_velocity[2]}

    def set_drone_params(self):
        drone_params = urdf.URDF.from_xml_file(f"{WORKING_DIRECTORY}/drone.urdf")
        self.mass = sum([m.inertial.mass for m in drone_params.links])

        motor_links = list(
            filter(lambda l: l.name.startswith("rotor_"), drone_params.links)
        )
        self.num_motors = len(motor_links)

        kv = float(drone_params.gazebos[0].attrib["kv"])
        voltage = float(drone_params.gazebos[0].attrib["voltage"])
        rpm_unloaded = kv * voltage
        self.max_rpm = rpm_unloaded * 0.75
        self.min_rpm = self.max_rpm * 0.04
        self.motor_scaling = self.max_rpm

    # pylint: disable=c-extension-no-member
    def step(self, action: Any) -> Tuple[ObsType, float, bool, dict]:
        physics_info = self._apply_physics(action)
        altitude = self._get_altitude()
        angles = np.clip(self._get_angles(), -1, 1)
        linear_velocity, _ = self.client.getBaseVelocity(self.drone_id)
        distance = self._get_distance()
        self.drone_img = self._get_drone_view()

        self.step_number = self.step_number + 1

        below_alt_min_limit = (
            self.step_number > 10 and altitude <= MIN_ALTITUDE / MAX_ALTITUDE
        )

        tilt_too_big = abs(angles[0]) > TILT_LITMIT or abs(angles[1]) > TILT_LITMIT

        above_alt_max_limit = altitude >= 1

        if self.use_gui is True:
            if above_alt_max_limit is True:
                print("Altitude is above limit -> episode ended")

            if below_alt_min_limit is True:
                print("Altitude is below limit -> episode ended")

            if tilt_too_big is True:
                print(f"Tilt too big {angles}")

        return (
            {
                "drone_img": self.drone_img,
                "distance": distance,
                "altitude": altitude,
                "roll": angles[0],
                "pitch": angles[1],
                "yaw": angles[2],
            },
            0,
            above_alt_max_limit or below_alt_min_limit or tilt_too_big,
            # above_alt_max_limit or below_alt_min_limit,
            False,
            {
                "step_number": self.step_number,
                "physics_info": physics_info,
                "vertical_velocity": linear_velocity[2],
            },
        )

    def render(self, mode="human"):
        return self.drone_img

    def close(self):
        pass
        # try:
        #     # self.client.disconnect()
        # except Exception as e:
        #      print(e)

    def _apply_physics(self, action: ActType):
        throttle, roll, pitch, yaw = action
        throttle = convert_range(throttle, -1, 1, 0, 1)
        # throttle = 0.26

        self.pid_pitch.setpoint = pitch
        self.pid_roll.setpoint = roll
        dt = 1.0 / 240.0
        angles = self._get_angles()
        roll_correction = self.pid_roll.compute(angles[0], dt)
        pitch_correction = self.pid_pitch.compute(angles[1], dt)

        rpms = np.array(
            [
                throttle + roll_correction - pitch_correction,  # - yaw, #top left
                throttle + roll_correction + pitch_correction,  # + yaw, # rear left
                throttle - roll_correction - pitch_correction,  # + yaw, # top right
                throttle - roll_correction + pitch_correction,  # - yaw, # rear right
            ]
        )

        rpms *= self.motor_scaling
        rpms = np.clip(rpms, 0, self.max_rpm)
        thrusts = self.thrust_coefficient * (rpms**2)
        total_torque = self.torque_coefficient * ((4 * yaw * self.motor_scaling) ** 2)

        for i in range(self.num_motors):
            self.client.applyExternalForce(
                self.drone_id, i, [0, 0, thrusts[i]], [0, 0, 0], p.LINK_FRAME
            )

        self.client.applyExternalTorque(
            self.drone_id, -1, [0, 0, total_torque], p.LINK_FRAME
        )
        self.client.stepSimulation()

        if self.use_gui is True:
            time.sleep(0.01)

        motor_positions = [
            [0.25, 0.25, -0.025],  # Motor 1
            [-0.25, 0.25, -0.025],  # Motor 2
            [0.25, -0.25, -0.025],  # Motor 3
            [-0.25, -0.25, -0.025],  # Motor 4
        ]

        for ind, pos in enumerate(motor_positions):
            start_pos = pos
            end_pos = pos[0], pos[1], pos[2] - thrusts[ind]

            # Add the debug line
            p.addUserDebugLine(
                start_pos, end_pos, [1, 0, 0], 2, parentObjectUniqueId=self.drone_id
            )  #

        return {
            "roll_actual": round(np.rad2deg(angles[0] * TILT_LITMIT), 4),
            "roll_target": round(np.rad2deg(roll * TILT_LITMIT), 4),
            "roll_correction": roll_correction,
            "pitch_actual": round(np.rad2deg(angles[1] * TILT_LITMIT), 4),
            "pitch_target": round(np.rad2deg(pitch * TILT_LITMIT), 4),
            "pitch_correction": pitch_correction,
        }

    def _get_distance(self) -> float:
        pos, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        rot_mat = self.client.getMatrixFromQuaternion(orn)
        drone_direction = np.array([rot_mat[0], rot_mat[3], rot_mat[6]]) * MAX_DISTANCE
        ray_result = self.client.rayTest(pos, pos + drone_direction)
        results = [hit[2] for hit in ray_result if hit[0] != -1]

        p.addUserDebugLine(
            [0, 0, 0],
            [MAX_DISTANCE, 0, 0],
            [0, 1, 0],
            2,
            0.1,
            parentObjectUniqueId=self.drone_id,
        )

        if len(results):
            return min(results)

        return 1

    def _get_angles(self) -> tuple[float, float, float]:
        _, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        orn_euler = self.client.getEulerFromQuaternion(orn)
        angles = [round(t, 4) for t in orn_euler]
        return [
            round(angles[0] / TILT_LITMIT, 4),
            round(angles[1] / TILT_LITMIT, 4),
            round(angles[2] / np.deg2rad(360), 4),
        ]

    def _get_altitude(self) -> float:
        pos, _ = self.client.getBasePositionAndOrientation(self.drone_id)

        return round(pos[2] / MAX_ALTITUDE, 4)

    def _get_drone_view(self) -> np.array:
        # pos, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        # rot_mat = np.array(self.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
        # target = np.dot(rot_mat, np.array([self.world_space.high[0], 0, 0])) + np.array(
        #     pos
        # )

        # drone_cam_view = self.client.computeViewMatrix(
        #     cameraEyePosition=pos, cameraTargetPosition=target, cameraUpVector=[0, 0, 1]
        # )
        # drone_cam_pro = self.client.computeProjectionMatrixFOV(
        #     fov=60.0, aspect=1.0, nearVal=0, farVal=np.max(self.world_space.high)
        # )
        # [width, height, rgb_img, dep, seg] = self.client.getCameraImage(
        #     width=256,
        #     height=256,
        #     shadow=1,
        #     viewMatrix=drone_cam_view,
        #     projectionMatrix=drone_cam_pro,
        # )

        # # Convert the image data to a numpy array
        # image = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))
        image = np.zeros((DRONE_IMG_HEIGHT, DRONE_IMG_WIDTH, 4))

        return image[:, :, :3]

    def _look_at(self, source_pos, target_pos):
        direction = np.array(target_pos) - np.array(source_pos)
        direction /= np.linalg.norm(direction)

        yaw = np.arctan2(direction[1], direction[0])
        pitch = np.arctan2(
            -direction[2], np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        )

        quat = self.client.getQuaternionFromEuler([0, pitch, yaw])
        return quat
