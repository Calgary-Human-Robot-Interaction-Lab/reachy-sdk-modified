"""Trajectory utility package.

Provides goto and goto_async functions. They let you easily create and compose movements on joint coordinates.
"""

import asyncio
import numpy as np
from scipy import signal

import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, Optional

from .interpolation import InterpolationMode
from ..joint import Joint


def goto(
    goal_positions: Dict[Joint, float],
    duration: float,
    starting_positions: Optional[Dict[Joint, float]] = None,
    sampling_freq: float = 100,
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
):
    """Send joints command to move the robot to a goal_positions within the specified duration.

    This function will block until the movement is over. See goto_async for an asynchronous version.

    The goal positions is expressed in joints coordinates. You can use as many joints target as you want.
    The duration is expressed in seconds.
    You can specify the starting_position, otherwise its current position is used,
    for instance to start from its goal position and avoid bumpy start of move.
    The sampling freq sets the frequency of intermediate goal positions commands.
    You can also select an interpolation method use (linear or minimum jerk) which will influence directly the trajectory.

    """

    exc_queue: Queue[Exception] = Queue()

    def _wrapped_goto():
        try:
            asyncio.run(
                goto_async(
                    goal_positions=goal_positions,
                    duration=duration,
                    starting_positions=starting_positions,
                    sampling_freq=sampling_freq,
                    interpolation_mode=interpolation_mode,
                ),
            )
        except Exception as e:
            exc_queue.put(e)

    with ThreadPoolExecutor() as exec:
        exec.submit(_wrapped_goto)
    if not exc_queue.empty():
        raise exc_queue.get()


async def goto_async(
    goal_positions: Dict[Joint, float],
    duration: float,
    starting_positions: Optional[Dict[Joint, float]] = None,
    sampling_freq: float = 100,
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
):
    """Send joints command to move the robot to a goal_positions within the specified duration.

    This function is asynchronous and will return a Coroutine. This can be used to easily combined multiple gotos.
    See goto for an blocking version.

    The goal positions is expressed in joints coordinates. You can use as many joints target as you want.
    The duration is expressed in seconds.
    You can specify the starting_position, otherwise its current position is used,
    for instance to start from its goal position and avoid bumpy start of move.
    The sampling freq sets the frequency of intermediate goal positions commands.
    You can also select an interpolation method use (linear or minimum jerk) which will influence directly the trajectory.

    """
    for key in goal_positions.keys():
        if not isinstance(key, Joint):
            raise ValueError('goal_positions keys should be Joint!')

    if starting_positions is None:
        starting_positions = {j: j.goal_position for j in goal_positions.keys()}

    # Make sure both starting and goal positions are in the same order
    starting_positions = {j: starting_positions[j] for j in goal_positions.keys()}

    length = round(duration * sampling_freq)
    if length < 1:
        raise ValueError('Goto length too short! (incoherent duration {duration} or sampling_freq {sampling_freq})!')

    joints = starting_positions.keys()
    dt = 1 / sampling_freq

    traj_func = interpolation_mode(
        np.array(list(starting_positions.values())),
        np.array(list(goal_positions.values())),
        duration,
    )

    t0 = time.time()
    while True:
        elapsed_time = time.time() - t0
        if elapsed_time > duration:
            break

        point = traj_func(elapsed_time)
        for j, pos in zip(joints, point):
            j.goal_position = pos

        await asyncio.sleep(dt)

glob_torque = {'l_shoulder_pitch' : 0, 'l_shoulder_roll' : 0, 'l_arm_yaw' : 0, 'l_elbow_pitch' : 0,
               'l_forearm_yaw' : 0, 'l_wrist_pitch' : 0, 'l_wrist_roll' : 0}

def get_torque(list_keys, t00):
    torque_condition = time.time() - t00
    #return {l : glob_torque[l] for l in list_keys}
    #return [glob_torque[l] for l in list_keys]
    
    #if torque_condition > 3 and torque_condition < 4:
    #    return  [0, 100, 0, 0, 100, 100, 0]
    #else:
    #    return np.zeros(len(list_keys))

    return np.zeros(len(list_keys))



def goto_compliant(
    damping_matrix,
    stiffness_matrix,
    goal_positions: Dict[Joint, float],
    duration: float,
    starting_positions: Optional[Dict[Joint, float]] = None,
    sampling_freq: float = 100,
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
):
    """Send joints command to move the robot to a goal_positions within the specified duration.

    This function will block until the movement is over. See goto_async for an asynchronous version.

    The goal positions is expressed in joints coordinates. You can use as many joints target as you want.
    The duration is expressed in seconds.
    You can specify the starting_position, otherwise its current position is used,
    for instance to start from its goal position and avoid bumpy start of move.
    The sampling freq sets the frequency of intermediate goal positions commands.
    You can also select an interpolation method use (linear or minimum jerk) which will influence directly the trajectory.

    """

    print("Entering goto function for modified library")

    exc_queue: Queue[Exception] = Queue()

    def _wrapped_goto_compliant():
        try:
            asyncio.run(
                goto_async_compliant(
                    damping_matrix=damping_matrix,
                    stiffness_matrix=stiffness_matrix,
                    goal_positions=goal_positions,
                    duration=duration,
                    starting_positions=starting_positions,
                    sampling_freq=sampling_freq,
                    interpolation_mode=interpolation_mode,
                ),
            )
        except Exception as e:
            exc_queue.put(e)

    with ThreadPoolExecutor() as exec:
        exec.submit(_wrapped_goto_compliant)
    if not exc_queue.empty():
        raise exc_queue.get()



async def goto_async_compliant(
    damping_matrix,
    stiffness_matrix,
    goal_positions: Dict[Joint, float],
    duration: float,
    starting_positions: Optional[Dict[Joint, float]] = None,
    sampling_freq: float = 100,
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
):
    """Send joints command to move the robot to a goal_positions within the specified duration.

    This function is asynchronous and will return a Coroutine. This can be used to easily combined multiple gotos.
    See goto for an blocking version.

    The goal positions is expressed in joints coordinates. You can use as many joints target as you want.
    The duration is expressed in seconds.
    You can specify the starting_position, otherwise its current position is used,
    for instance to start from its goal position and avoid bumpy start of move.
    The sampling freq sets the frequency of intermediate goal positions commands.
    You can also select an interpolation method use (linear or minimum jerk) which will influence directly the trajectory.

    """

    for key in goal_positions.keys():
        if not isinstance(key, Joint):
            raise ValueError('goal_positions keys should be Joint!')

    if starting_positions is None:
        starting_positions = {j: j.goal_position for j in goal_positions.keys()}

    # Make sure both starting and goal positions are in the same order
    starting_positions = {j: starting_positions[j] for j in goal_positions.keys()}

    def get_current_pos(goal_positions):
        current_positions = {j: j.present_position for j in goal_positions.keys()}
        current_positions = {j: current_positions[j] for j in goal_positions.keys()}
        return current_positions
    

    length = round(duration * sampling_freq)
    if length < 1:
        raise ValueError('Goto length too short! (incoherent duration {duration} or sampling_freq {sampling_freq})!')

    joints = starting_positions.keys()
    dt = 1 / sampling_freq

    gpk = list(goal_positions.keys())
    joint_list = [x.name for x in gpk]
    #torque_list = get_torque(joint_list)

    print("Goal Positions: ", goal_positions)
    print("Goal Positions Keys: ", gpk)
    print("Joint List: ", joint_list)
    #print("Torque List: ", torque_list)

    traj_func = interpolation_mode(
        np.array(list(starting_positions.values())),
        np.array(list(goal_positions.values())),
        duration,
    )

    B = np.linalg.inv(damping_matrix)
    A = -1 * np.matmul(B, stiffness_matrix)

    #B_df = dt * B
    #A_df = ((np.identity(len(A))) + (dt * A))

    A_df = np.linalg.inv(((np.identity(len(A))) - (dt * A)))
    B_df = dt * np.matmul(A_df, B)
    
    #C_df = np.identity(len(A_df))
    #D_df = np.zeros(A_df.shape)

    #sys = signal.StateSpace(A_df, B_df, C_df, D_df, dt = dt)

    x_k = np.zeros(len(A_df))
    
    # Commenting with caution
    #mod_duration = duration
    
    prev_pos = get_current_pos(goal_positions)
    prev_pos = np.array(list(prev_pos.values()))

    t0 = time.time()
    t00 = time.time()
    while True:
        elapsed_time = time.time() - t0
        #if elapsed_time > duration:
        #    break

        current_positions = get_current_pos(goal_positions)
        current_positions_vals = np.array(list(current_positions.values()))
        goal_positions_vals = np.array(list(goal_positions.values()))
        
        u_k = get_torque(joint_list, t00)

        error = np.amax(np.abs(np.subtract(goal_positions_vals, current_positions_vals)))
        change = np.amax(np.abs(np.subtract(current_positions_vals, prev_pos)))
        
        # Tunable Parameters
        err_lim = 5
        change_lim = 0.1

        #print("elapsed time: ", elapsed_time)
        #print("torque: ", u_k)
        #print("current_positions_vals: ", current_positions_vals)
        #print("goal_positions_vals: ", goal_positions_vals)
        #print("error: ", error, "\n\n")


        if error < err_lim and change < change_lim:
            print("Breaking")
            print("elapsed_time", elapsed_time)
            print("Goal Positions", goal_positions_vals)
            print("Final Position: ", current_positions_vals)
            print("Error: ", error)
            print("Change: ", change, "\n")
            break


        # Tunable Parameters
        torque_lim = 100

        #print(elapsed_time)
        if np.sum(u_k) > torque_lim:
            print("Torque Detected")
            t0 = time.time()

            # Commenting with caution
            #mod_duration = duration + dt
            
            traj_func = interpolation_mode(
                np.array(list(current_positions.values())),
                np.array(list(goal_positions.values())),
                duration,
            )
        
        # Commenting with caution
        # mod_duration_loop = mod_duration - elapsed_time

        x_k_1 = np.matmul(A_df, x_k) + np.matmul(B_df, u_k)
        #if np.sum(x_k_1) > 0:
        #    print(x_k_1)


        point = traj_func(elapsed_time)
        for j, pos, adm in zip(joints, point, x_k_1):
            #j.goal_position = pos
            j.goal_position = pos + adm 

        x_k = x_k_1
        prev_pos = current_positions_vals

        await asyncio.sleep(dt)