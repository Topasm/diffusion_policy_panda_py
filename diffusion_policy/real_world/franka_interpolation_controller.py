import numpy as np
import franky
import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class FrankaInterface:
    def __init__(self, ip='172.16.0.2'):
        self.robot = franky.Robot(ip)
        self.robot.relative_dynamics_factor = 0.3
        self.robot.recover_from_errors()

    def get_ee_pose(self):
        current_pose = self.robot.current_cartesian_state.pose
        ee_pose = current_pose.end_effector_pose

        # Assuming correct attribute names based on inspection
        translation = np.array(ee_pose.translation)
        quaternion = np.array(ee_pose.quaternion)
        rotvec = st.Rotation.from_quat(quaternion).as_rotvec()

        return np.concatenate((translation, rotvec))

    def get_joint_positions(self):
        return np.array(self.robot.current_joint_state.position)

    def get_joint_velocities(self):
        return np.array(self.robot.current_joint_state.velocity)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):

        motion = franky.JointWaypointMotion(
            [franky.JointWaypoint(positions.tolist())])
        self.robot.move(motion)

    def start_cartesian_impedance(self, Kx: np.ndarray):
        self.robot.set_cartesian_impedance(Kx.tolist())

    def update_desired_ee_pose(self, pose: np.ndarray):

        translation = pose[:3].reshape(3, 1)
        quaternion = st.Rotation.from_rotvec(pose[3:]).as_quat()

        affine = franky.Affine(translation, quaternion)
        motion = franky.CartesianMotion(franky.RobotPose(
            affine), reference_type=franky.ReferenceType.Absolute)

        # print('translation', translation)
        # print('quaternion', quaternion)

        self.robot.move(motion, asynchronous=True)

    def close(self):
        self.robot.stop()
        pass


class FrankaInterpolationController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 frequency=200,
                 launch_timeout=3,
                 joints_init=None,
                 joints_init_duration=None,
                 soft_real_time=False,
                 verbose=True,
                 get_max_k=None,
                 receive_latency=0.0):
        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd', 'get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)  # Change to 6 for euler angles

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    def start(self, wait=True):
        super().start()
        if wait:
            print('wait')
        if self.verbose:
            print(
                f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {'cmd': Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            print('wait')

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run_memstate(self):
        robot = FrankaInterface(self.robot_ip)
        state = dict()
        for key, func_name in self.receive_keys:
            # Call the function
            state[key] = getattr(robot, func_name)()

        t_recv = time.time()
        state['robot_receive_timestamp'] = t_recv
        state['robot_timestamp'] = t_recv - self.receive_latency
        self.ring_buffer.put(state)
        # print('run_memstate', state)

    def run(self):
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        robot = FrankaInterface(self.robot_ip)

        try:
            if self.verbose:
                print(
                    f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")

            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=self.joints_init, time_to_go=self.joints_init_duration)

            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t], poses=[curr_pose[:6]])  # Ensure the pose has the correct shape

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                tip_pose = pose_interp(t_now)

                # print('tip_pose', tip_pose)

                robot.update_desired_ee_pose(tip_pose)

                state = dict()
                for key, func_name in self.receive_keys:
                    # Call the function
                    state[key] = getattr(robot, func_name)()

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(
                        f"[FrankaPositionalController] Actual frequency {1 / (time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.close()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(
                    f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
