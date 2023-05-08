import os
import argparse
import pathlib
import numpy as np
import imageio
from PIL import Image


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

def eplen(episode):
    return len(episode['action'])


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='robosuite_expert_small/Lift/Panda/OSC_POSE', type=str)
args = parser.parse_args()

directory = pathlib.Path(args.path)
# print(directory)
episodes = load_episodes(directory)

print(f"Loaded {len(episodes)} episodes")
print(f"Keys and value shapes:")
filenames = sorted(directory.glob('*.npz'))

# print(len(filenames))
# for filename in filenames:
#     print(filename, episodes[str(filename)].keys())
#     break

    # np.savez(filename, episodes[str(filename)])
    # for k, v in episodes[str(filename)].items():
    #     # print(k, v.shape)

    #     if k == 'robot0_touch-state':
    #         print(k, np.unique(v))
        
    #     if k == 'robot0_touch':
    #         print(k, np.unique(v))


# stitch the images in an episode from expert data and store in a video for visualization
print("Visualize the expert data images")
for filename in filenames:
    # print(filename)
    # print(episodes[str(filename)])
    episode = episodes[str(filename)]
    print(episode.keys())
    images = episode['agentview_image']
    rewards = episode['reward']
    actions = episode['action']
    print(images.shape, images.min(), images.max())
    print(actions.shape, actions.min(), actions.max())
    # print(rewards.shape, rewards)
    
    
    # video_path = "robosuite_expert_small/videos/test.mp4"
    # video_writer = imageio.get_writer(video_path, fps=30)
    # frames = episodes[str(filename)]['agentview_image']
    # for i in range(1001):
    #     frame = np.array(Image.fromarray(frames[i]).resize((512, 512)))
    #     video_writer.append_data(frame)
    # video_writer.close()
    # break