import imageio
import os
import numpy as np
from PIL import Image

# import robosuite.macros as macros
# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio (which uses opencv convention)
# macros.IMAGE_CONVENTION = "opencv"


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, render_camera_names=["frontview"], fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        # self.camera_id = camera_id
        self.render_camera_names = render_camera_names
        self.fps = fps
        # self.frames = []

    def init(self, enabled=True, filename=None):
        # self.frames = []
        self.enabled = self.dir_name is not None and enabled

        if self.enabled:
            print("Video Enabled")
            self.video_writer = []
            
            for i in range(len(self.render_camera_names)):
                video_path = os.path.join(self.dir_name, filename.split('.')[0]+"_"+self.render_camera_names[i]+"."+filename.split('.')[1])
                self.video_writer.append(imageio.get_writer(video_path, fps=self.fps))

    def record(self, env, obs):
        if self.enabled:
            # frames = np.transpose(obs, (1,2,0)).astype(np.uint8)
            # frame = np.array(Image.fromarray(frames[:,:,0:3]).resize(size=(self.height, self.width)))
            # self.frames.append(frame)

            for i in range(len(self.render_camera_names)):
                frame = env.sim.render(height=self.height, width=self.width, camera_name=self.render_camera_names[i])[::-1]
                self.video_writer[i].append_data(frame)

    def save(self, file_name):
        if self.enabled:
            for i in range(len(self.render_camera_names)):
                self.video_writer[i].close()
            # path = os.path.join(self.dir_name, file_name)
            # imageio.mimsave(path, self.frames, fps=self.fps)
