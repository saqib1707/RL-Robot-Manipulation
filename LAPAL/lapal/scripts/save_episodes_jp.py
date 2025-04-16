import datetime
import uuid
import io
import pathlib
import numpy as np
import robosuite as suite

from lapal.utils import utils
from human_policy import LiftPolicy, ReachPolicy

"""
Solve the task with osc_pose controller, and save the joint position observations along the trajectory
Replay the joint position 
"""

def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename

def eplen(episode):
    return len(episode['action'])

def load_episodes(directory, capacity=None):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob('*.npz'))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split('-')[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        episodes[str(filename)] = episode
    return episodes

def osc_to_jp(directory, render=False):

    # Intialize an env with OSC_POSE controller
    env = utils.make_robosuite_env("Lift", render=render)
    policy = LiftPolicy(env)

    obs = env.reset()

    # Record initial state since it is the same when replayed
    obs_keys = list(obs.keys())
    episode = {}
    for k in obs_keys:
        episode[k] = [obs[k]]
    episode['action'] = []
    episode['reward'] = []

    # Record the mujoco states so that we can replay later
    task_xml = env.sim.model.get_xml()
    task_init_state = np.array(env.sim.get_state().flatten())

    policy.reset()
    waypoints = []

    done = False
    while not done:
        action, action_info = policy.predict(obs)
        if action_info['completed'] and action_info['stage'] != 'lift_gripper':
            waypoints.append((env.robots[0]._joint_positions, action[-1], action_info))        

        obs, rew, done, info = env.step(action)

        if render:
            env.render()


    waypoints.append((env.robots[0]._joint_positions, action[-1], action_info))  
    env.close()

    # Initialize an env with JOINT_POSITION controller
    # Make sure controller input and output ranges match (without scaling)
    env = utils.make_robosuite_env("Lift", controller_type="JOINT_POSITION", render=render)

    # Reset environment to the same initial state
    env.reset()
    env.reset_from_xml_string(task_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(task_init_state)
    env.sim.forward()     


    gripper_action = -1
    done = False
    waypoints = iter(waypoints)
    waypoint = next(waypoints)
    while not done:
        action = np.zeros(env.robots[0].dof)
        action[-1] = gripper_action

        if waypoint == 'end':
            action[-1] = 1

        else:
            jp, gripper_action, action_info = waypoint
            error, tol = 1, 2e-2

            if action_info['stage'] == 'rotate_z':
                error = obs[f'robot0_eef_to_cube_yaw']
                tol = 5e-2
            elif action_info['stage'] == 'move_x':
                error = -obs[f'cube_to_robot0_eef_pos'][0]
            elif action_info['stage'] == 'move_y':
                error = -obs[f'cube_to_robot0_eef_pos'][1]
            elif action_info['stage'] == 'lower_gripper':
                error = -obs[f'cube_to_robot0_eef_pos'][2]
            elif action_info['stage'] == 'close_gripper':
                error = 1 - obs['robot0_touch']
                tol = 0.05
            elif action_info['stage'] == 'lift_gripper':
                error = 0.9 - obs['robot0_eef_pos'] 

            if np.linalg.norm(error) < tol:
                waypoint = next(waypoints, 'end')

            action = np.zeros(env.robots[0].dof)
            # Multiply by 20 to account for controller input/output scaling
            jp_action = np.clip((jp - env.robots[0]._joint_positions) * 20, -1, 1)
            action[:-1] = jp_action
            action[-1] = gripper_action
        obs, rew, done, info = env.step(action)
            
        for k in obs_keys:
            episode[k].append(obs[k])
        episode['action'].append(action)
        episode['reward'].append(rew)

        if render:
            env.render()

        if done:
            env.close()
            return episode
    assert False, "Should not end here"


def osc_to_jp_Reach(directory, render=False):

    robots = "Panda"
    # robots = "UR5e"
    # robots = "Sawyer"
    # Intialize an env with OSC_POSE controller
    env = utils.make_robosuite_env("Reach", robots=robots, render=render)
    policy = ReachPolicy(env)

    obs = env.reset()
    target_pos = np.copy(env.target_pos)

    # Record initial state since it is the same when replayed
    obs_keys = list(obs.keys())
    episode = {}
    for k in obs_keys:
        episode[k] = [obs[k]]
    episode['action'] = []
    episode['reward'] = []

    # Record the mujoco states so that we can replay later
    task_xml = env.sim.model.get_xml()
    task_init_state = np.array(env.sim.get_state().flatten())

    policy.reset()
    waypoints = []

    done = False
    while not done:
        action, action_info = policy.predict(obs)
        if action_info['completed']:
            waypoints.append((env.robots[0]._joint_positions, action_info))        

        obs, rew, done, info = env.step(action)

        if render:
            env.render()


    waypoints.append((env.robots[0]._joint_positions, action_info))  
    env.close()

    # Initialize an env with JOINT_POSITION controller
    # Make sure controller input and output ranges match (without scaling)
    env = utils.make_robosuite_env("Reach", robots=robots, controller_type="JOINT_POSITION", render=render)

    # Reset environment to the same initial state
    env.reset(target_pos=target_pos)
    env.reset_from_xml_string(task_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(task_init_state)
    env.sim.forward()     

    gripper_action = -1
    done = False
    waypoints = iter(waypoints)
    waypoint = next(waypoints)
    while not done:
        action = np.zeros(env.robots[0].dof)
        action[-1] = gripper_action

        if waypoint == 'end':
            action[-1] = gripper_action

        else:
            jp, action_info = waypoint
            error, tol = 1, 2e-2

            if action_info['stage'] == 'move_x':
                error = -obs[f'target_to_robot0_eef_pos'][0]    
            elif action_info['stage'] == 'move_y':
                error = -obs[f'target_to_robot0_eef_pos'][1]    
            elif action_info['stage'] == 'move_z':
                error = -obs[f'target_to_robot0_eef_pos'][2]    

            if np.linalg.norm(error) < tol:
                waypoint = next(waypoints, 'end')

            action = np.zeros(env.robots[0].dof)
            # Multiply by 20 to account for controller input/output scaling
            jp_action = np.clip((jp - env.robots[0]._joint_positions) * 20, -1, 1)
            action[:-1] = jp_action
        obs, rew, done, info = env.step(action)
            
        for k in obs_keys:
            episode[k].append(obs[k])
        episode['action'].append(action)
        episode['reward'].append(rew)

        if render:
            env.render()

        if done:
            env.close()
            return episode

    
    assert False, "Should not end here"

def main():
    directory = pathlib.Path(f"./human_demonstrations/Reach/Panda/JOINT_POSITION")

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
    episodes_saved = 0
    while episodes_saved < 64:
        episode = osc_to_jp_Reach(directory, render=False)
        print(f"Episode return: {np.sum(episode['reward']):.2f}")
        if np.sum(episode['reward']) < 100:
            print("Discarding current episode")
        else:
            save_episode(directory, episode)
            episodes_saved += 1

if __name__ == '__main__':
    main()