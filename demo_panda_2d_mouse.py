"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_panda_2d import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
# from diffusion_policy.real_world.keystroke_counter import (
#     KeystrokeCounter, Key, KeyCode
# )

from skill_utils.mouse_teleop import MouseTeleop
from skill_utils.teleop import KeyboardCommandHandler

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="10.0.0.2", help="Panda's IP address e.g. 10.0.0.2")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1/frequency
    
    with SharedMemoryManager() as shm_manager:
        with RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(640, 480),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=False,
                record_raw_video=False,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
        ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)
            time.sleep(1.0)

            currentpose=np.reshape(env.get_robot_state()['ActualTCPPose'],(4,4)).T
            print(f"Setting pose to {currentpose}")
            
            with MouseTeleop(currentpose) as mouse, KeyboardCommandHandler() as kb:
                print('Ready!')
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                stop = False
                is_recording = False
                while not stop:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # handle key presses
                    if kb.endevent.is_set():
                        # Exit program
                        kb.endevent.clear()
                        stop = True
                    elif kb.startevent.is_set():
                        # Start recording
                        kb.startevent.clear()
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        is_recording = True
                        print('Recording!')
                    elif kb.stopevent.is_set():
                        # Stop recording
                        kb.stopevent.clear()
                        env.end_episode()
                        is_recording = False
                        print('Stopped.')
                    elif kb.delevent.is_set():
                        # Delete the most recent recorded episode
                        kb.delevent.clear()
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            is_recording = False
                    stage=kb.stagecounter
                    
                    # visualize
                    # vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                    # episode_id = env.replay_buffer.n_episodes
                    # text = f'Episode: {episode_id}, Stage: {stage}'
                    # if is_recording:
                    #     text += ', Recording!'
                    # cv2.putText(
                    #     vis_img,
                    #     text,
                    #     (10,30),
                    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #     fontScale=1,
                    #     thickness=2,
                    #     color=(255,255,255)
                    # )

                    # cv2.imshow('default', vis_img)

                    precise_wait(t_sample)
                    # get teleop command
                    target_pose = mouse.formatted_pose

                    # execute teleop command
                    env.exec_actions(
                        actions=[target_pose], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        stages=[stage])
                    precise_wait(t_cycle_end)
                    iter_idx += 1

# %%
if __name__ == '__main__':
    main()
