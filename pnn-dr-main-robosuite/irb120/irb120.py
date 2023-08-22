import os
import mujoco_py
# import mujoco
import numpy as np
from gym.utils import seeding

DEFAULT_SIZE = 64


class IRB120Env:
    def __init__(
        self, width, height, frame_skip, rewarding_distance, control_magnitude, reward_continuous, max_ep_length
    ):
        """
        Class constructor.

        Args:
            width (int): image width.
            height (int): image height.
            frame_skip (int): frame skipping in environment. Repeats last agent action.
            rewarding_distance (float): distance to target at which positive reward is obtained.
            control_magnitude (float): fraction of actuator range used as control inputs.
            reward_continuous (bool): if True, provides rewards at every timestep.
            max_ep_length (int): maximum episode length.

        Raises:
            IOError: if the file with the model does not exist
        """
        # Viewer and image parameters
        self.frame_skip = frame_skip
        self.width = width
        self.height = height
        self.viewer = None
        self._viewers = {}
        # Robot parameters
        model_path = "irb120.xml"
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        model = mujoco_py.load_model_from_path(fullpath)
        # model = mujoco.mj_loadXML(fullpath)
        self.sim = mujoco_py.MjSim(model)
        # self.sim = mujoco.MjSim(model)
        self.init_state = self.sim.get_state()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.actuator_bounds = self.sim.model.actuator_ctrlrange
        self.actuator_low = self.actuator_bounds[:, 0]
        self.actuator_high = self.actuator_bounds[:, 1]
        self.actuator_ctrlrange = self.actuator_high - self.actuator_low
        self.num_actuators = len(self.actuator_low)
        self.sim.data.ctrl[:] = np.zeros(self.num_actuators)
        # MDP and problem parameters
        self.seed()
        self.reward_continuous = reward_continuous
        self.sum_reward = 0
        self.rewarding_distance = rewarding_distance
        self.max_threshold = 0.3
        self.target_bounds = np.array(((0.2, 0.4), (-0.3, 0.3), (0.02, 0.02)))
        self.target_reset_distance = 0.1
        self.control_values = self.actuator_ctrlrange * control_magnitude
        self.num_actions = 7
        self.action_space = [list(range(self.num_actions))] * self.num_actuators
        self.observation_space = ((0,), (height, width, 3))
        self.max_ep_length = max_ep_length
        self.episode_end = 0
        # For the original view, set self.deg_prev to 180
        self.deg_prev = np.random.randint(160, 201)
        # For the original view, set self.elevation_prev to -30
        self.elevation_prev = np.random.randint(-40, -21)
        # Reset the environment
        self.reset()

    def seed(self, seed=None):
        """
        Set a random environment seed.

        Args:
            seed (int, optional): passed seed. Defaults to None.

        Returns:
            list: seed value
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_qpos_qvel(self, qpos, qvel):
        """
        Set the joints' position and velocity.

        Args:
            qpos (np.ndarray): joints' position.
            qvel (np.ndarray): joints' velocity.
        """
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    def reset(self):
        """
        Reset the environment.

        Returns:
            tuple: state observation with the joints' position and the env image.
        """
        qpos = self.init_qpos
        # Reset initial positions
        # Range: 15% of its total range since 20% involves positions not showned on the window
        qpos[:1] = np.round(np.random.uniform(-0.44, 0.44, 1), 2)
        qpos[1:2] = np.round(np.random.uniform(-0.29, 0.29, 1), 2)
        qvel = self.init_qvel
        self._reset_target()
        self._set_qpos_qvel(qpos, qvel)
        return self._get_obs()

    def _reset_target(self):
        """
        Randomize goal position within specified bounds.
        """
        self.goal = np.random.rand(3) * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]
        geom_positions = self.sim.model.geom_pos.copy()
        prev_goal_location = geom_positions[1]
        while np.linalg.norm(prev_goal_location - self.goal) < self.target_reset_distance:
            self.goal = (
                np.random.rand(3) * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]
            )
        geom_positions[1] = self.goal
        self.sim.model.geom_pos[:] = geom_positions

    def render(self, mode=None, width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_id=None, camera_name=None):
        """
        Render the environment.

        Args:
            mode (str, optional): render mode for visualization. Defaults to None.
            width (int, optional): image width. Defaults to DEFAULT_SIZE.
            height (int, optional): image height. Defaults to DEFAULT_SIZE.
            camera_id (int, optional): camera id set on the .xml file. Defaults to None.
            camera_name (str, optional): camera name set on the .xml file. Defaults to None.

        Raises:
            ValueError: if both camera_id and camera_name are provided.

        Returns:
            np.ndarray: camera observation of the environment.
        """
        if mode == "rgb_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both 'camera_id' and 'camera_name' cannot be" " specified at the same time.")
            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"
            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)
            self._get_viewer(mode).render(width, height, camera_id=-1)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # Original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        """
        Obtain the viewer.

        Args:
            mode (str): render mode for visualization.

        Returns:
            mujoco_py.cymj.MjRenderContext: viewer for the environment.
        """
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
                # self.viewer = mujoco.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1, opengl_backend="glfw")
                # self.viewer = mujoco.MjRenderContextOffscreen(self.sim, -1, opengl_backend="glfw")
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        self._pose_cam()
        return self.viewer

    def _viewer_setup(self):
        """
        Viewer setup.
        """
        body_id = self.sim.model.body_name2id("robot0:base_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

    def _pose_cam(self):
        """
        Camera pose.
        """
        self.viewer.cam.distance = 2
        # For the original view, set self.elevation to -30
        elevation = np.random.randint(-40, -21)
        # For the original view, set deg to 180
        deg = np.random.randint(160, 201)
        if self.episode_end == 1:
            self.viewer.cam.elevation = elevation
            self.viewer.cam.azimuth = deg
            self.deg_prev = deg
            self.elevation_prev = elevation
            self.episode_end = 0
        elif self.episode_end == 0:
            self.viewer.cam.elevation = self.elevation_prev
            self.viewer.cam.azimuth = self.deg_prev
        return

    def _get_obs_joint(self):
        """
        Get the observation of the joints' position and velocity.

        Returns:
            np.ndarray: joints' position and velocity.
        """
        return np.concatenate([self.sim.data.qpos.flat[:], self.sim.data.qvel.flat[:]])

    def _get_obs_rgb_view1(self):
        """
        Get the image observation.

        Returns:
            np.ndarray: camera observation of the environment.
        """
        self._get_viewer("rgb_array").render(self.width, self.height, camera_id=-1)
        obs_rgb_view1 = self._get_viewer("rgb_array").read_pixels(self.width, self.height, depth=False)
        return obs_rgb_view1[::-1, :]

    def _get_obs(self):
        """
        Get the complete observation.

        Returns:
            tuple: variable observation and image observation.
        """
        return (self._get_obs_joint(), self._get_obs_rgb_view1())

    def _do_simulation(self, ctrl):
        """
        Do one step of simulation, taking new control as target.

        Arguments:
            ctrl (np.array(num_actuator)): new control to send to actuators.
        """
        self.sim.data.ctrl[:] = ctrl
        # Gravity compensation
        self.sim.data.qfrc_applied[:] = self.sim.data.qfrc_bias[:]
        for _ in range(self.frame_skip):
            self.sim.step()

    def step(self, a, ep_length):
        """
        Performs a single step simulation.

        Args:
            a (list): action that each joint must perform.
            ep_length (int): number of current steps in the episode.

        Returns:
            tuple, float, bool: the complete observation, the reward for this step, if the episode is completed or not.
        """
        done = False
        new_control = np.copy(self.sim.data.ctrl).flatten()
        inc_rad = np.copy(self.sim.data.ctrl).flatten()
        dist = np.linalg.norm(self.sim.data.get_site_xpos("robot0:grip") - self.goal)
        # Check if the robot has reached the desired area and set its reward
        if dist <= self.rewarding_distance and self.sim.data.get_site_xpos("robot0:grip")[2] >= 0.005:
            reward = 70
            done = True
        # Continuous reward if the goal is not reached
        if self.reward_continuous and done == False:
            reward = -((2 * dist) ** 2)
        # Sparse reward if the goal is not reached
        elif not self.reward_continuous:
            if dist > self.rewarding_distance:
                reward = -1
            else:
                reward = 0
        ## Transform discrete actions to continuous controls
        for i in range(self.num_actuators):
            if a[i] == 0:
                inc_rad[i] = 0
            if a[i] == 1:
                inc_rad[i] = self.control_values[i] / 100
            if a[i] == 2:
                inc_rad[i] = self.control_values[i] / 10
            if a[i] == 3:
                inc_rad[i] = self.control_values[i]
            if a[i] == 4:
                inc_rad[i] = -self.control_values[i] / 100
            if a[i] == 5:
                inc_rad[i] = -self.control_values[i] / 10
            if a[i] == 6:
                inc_rad[i] = -self.control_values[i]
        if done == True or ep_length == self.max_ep_length - 1:
            self.episode_end = 1
        new_control = self.sim.data.qpos[0:6] + inc_rad
        # Do one step of simulation
        self._do_simulation(new_control)
        self.sum_reward += reward
        return self._get_obs(), reward, done
