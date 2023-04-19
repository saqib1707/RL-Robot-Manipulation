from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

max_frame = 90

width = 480
height = 480
video = np.zeros((90, height, 2 * width, 3), dtype=np.uint8)

# Load one task:
# env = suite.load(domain_name="cartpole", task_name="swingup")
env = suite.load(domain_name="cheetah", task_name="run")

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
# while not time_step.last():
for _ in tqdm(range(10)):
  for i in range(max_frame):
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    # print("Before physics render")
    cam0_render = env.physics.render(height, width, camera_id=0)
    cam1_render = env.physics.render(height, width, camera_id=1)
    # print("After physics render")
    video[i] = np.hstack([cam0_render, cam1_render])
    #print(time_step.reward, time_step.discount, time_step.observation)
#   for i in range(max_frame):
#     img = plt.imshow(video[i])
#     plt.pause(0.01)  # Need min display time > 0.0.
#     plt.draw()

print("Test successful !!!")