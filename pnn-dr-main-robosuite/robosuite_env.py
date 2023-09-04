import numpy as np
import torch
from tqdm import tqdm
import time
import imageio

import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.wrappers import DomainRandomizationWrapper

# from robosuite.utils.mjmod import CameraModder  
# from robosuite.utils.camera_utils import CameraMover

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
    # print('Allocated memory:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
    # print('Cached memory:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
else:
    print("Device:", device)

# print robosuite welcome info
print("Welcome to robosuite v{}!".format(suite.__version__))
print(suite.__logo__)


env_name = "Lift"
controller_config = load_controller_config(default_controller="OSC_POSE")  # load default controller parameters for Operational Space Control (OSC)
# controller_config = None
train_camera_names = ["agentview", "robot0_eye_in_hand", "frontview", "bestview"]
horizon = 100
image_size = 512
use_camera_depth = False
use_touch_obs = True
use_tactile_obs = False


# create an environment to visualize on-screen
env = suite.make(
    env_name=env_name,
    robots="Panda",                # load a Sawyer robot and/or a Panda robot
    gripper_types="default",         # use default grippers for robot arm
    controller_configs=controller_config,      # each arm is controlled using OSC
    # env_configuration="single-arm-opposed",    # arms face each other
    reward_shaping=True, 
    has_renderer=True,                         # on-screen rendering
    # render_camera="frontview",                 # visualize the frontview camera
    has_offscreen_renderer=True,              # no off-screen rendering
    control_freq=20,                    
    horizon=horizon,             # each episode terminates after 'horizon' steps
    use_object_obs=False,        # no observations needed
    use_camera_obs=True,         # don't provide camera/image observations to agent
    camera_depths=use_camera_depth,
    camera_heights=image_size, 
    camera_widths=image_size, 
    camera_names=train_camera_names, 
    use_touch_obs=use_touch_obs, 
    use_tactile_obs=use_tactile_obs,
    ignore_done=True,
    hard_reset=False,  # TODO: Not setting this flag to False brings up a segfault on macos or glfw error on linux
)

# Wrapper that allows for domain randomization mid-simulation.
env = DomainRandomizationWrapper(
    env, 
    seed=np.random.randint(0,100),
    randomize_color=True,       # if True, randomize geom colors and texture colors
    randomize_camera=True,      # if True, randomize camera locations and parameters
    randomize_lighting=True,    # if True, randomize light locations and properties
    randomize_dynamics=True,    # if True, randomize dynamics parameters
    randomize_on_reset=True, 
    randomize_every_n_steps=5
)

# modder_obj = CameraModder(env)
# modder_obj.set_pos(
#     "frontview",
#     modder_obj._defaults["frontview"]["pos"] + np.array([0.1,0.1,0.1]),
# )

# camera_mover = CameraMover(env)
# camera_mover.set_camera_pose(pos=np.array([100.0,100.0,100.0]))

obs = env.reset()
for k, v in obs.items():
    print(k, v.shape)

# env.viewer.set_camera(camera_id=0)
low, high = env.action_spec        # Get action limits

total_reward = 0
# frames_rgb = []

# create a video writer with imageio
video_path = "visualizations/view_rgb.mp4"
writer = imageio.get_writer(video_path, fps=20)

start_time = time.time()

# do visualization
for _ in range(1):
    obs = env.reset()
    for i in tqdm(range(horizon)):
        # print("iteration:", i)
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        # env.render()
        # frames_rgb.append(obs["frontview_image"])
        writer.append_data(obs["bestview_image"])
    
writer.close()

print("rollout completed with return {}".format(total_reward))
print(f"Spend {time.time() - start_time:.3f} s to run {horizon} steps")

# video_path = "visualizations/view_rgb.mp4"
# imageio.mimsave(video_path, frames_rgb, fps=30)

env.close()