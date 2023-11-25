## Useful Resources
- SAC-AE: [code](https://github.com/denisyarats/pytorch_sac_ae)
- LAPAL: [code](https://github.com/tianyudwang/LAPAL)
- VMAIL: [code](https://github.com/rmrafailov/VMAIL)
- MuJoCo (Multi-Joint Dynamics with Contact): [code](https://github.com/openai/mujoco-py)
- [Robosuite Domain Randomization](https://robosuite.ai/docs/source/robosuite.wrappers.html)
- [Nautilus setup guide]()


**To add git repository to the project in VS code:**
```
git config --global --add safe.directory /path/to/project/directory
```


## Nautilus Cluster

### Available GPUs
- NVIDIA A10
- NVIDIA A100-SXM4-80GB (consumes much more memory space compared to other GPUs)
- NVIDIA A100-PCIE-40GB
- Quadro RTX 6000
- NVIDIA-GeForce-GTX-1080-Ti
- NVIDIA-GeForce-RTX-2080-Ti
- NVIDIA-GeForce-RTX-3090
- NVIDIA A100 80GB PCIe MIG 1g.10gb
- NVIDIA TITAN-RTX

### Kubernetes commands
Below are some commonly used Kubernetes commands along with brief descriptions:
##### 1. Get Information about Pods
```
kubectl get pods
```

##### 2. Fetch information about the cluster nodes, specifically looking for nodes with NVIDIA GPUs and their corresponding GPU models.
```
kubectl get nodes -L nvidia.com/gpu.product
```

##### 3. Create a new Kubernetes job based on the configuration provided. Jobs are used for running batch processes or scheduled tasks.
```
kubectl create -f create_job_config.yaml
```

##### 4. Execute a command interactively within a specific pod
```
kubectl exec -it <pod name> -- /bin/bash
```

##### 5. Delete a specific pod
```
kubectl delete pod <pod name>
```

##### 5. Deletes a specific job which terminates its associated pods and frees up any allocated resources
```
kubectl delete job <job name>
```

##### 6. Retrieve information about Persistent Volume Claims (PVCs) in the current namespace. Displays details such as PVC names, statuses, and storage capacity. PVCs are used to request persistent storage in a Kubernetes cluster.
```
kubectl get pvc
```


## Robosuite Environment

- Install official robosuite using python pip
```
pip3 install --no-cache-dir robosuite
```

**Note:** 
Might face an error due to PyOpenGL version. PyOpenGL v3.1.4 is not compatible whereas v3.1.6 is compatible. Do not install PyOpenGL separately using pip. Installing robosuite with --no-cache-dir should automatically install PyOpenGL as a dependency.

- Install Tianyu’s robosuite branch
```
git clone https://github.com/tianyudwang/robosuite.git
cd robosuite && pip3 install -e .
```

**Note:** Ensure PyOpenGL version is v3.1.4 and not v3.1.0. Otherwise you may get the following error `“AttributeError: module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT'”`

**Trick:** Set the image convention to `opencv` in the python script when importing robosuite so that the images are automatically rendered "right side up" when using imageio (which uses opencv convention). 
```
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
```

### Registered Environments
- Lift
- Stack
- NutAssembly
- NutAssemblySingle
- NutAssemblySquare
- NutAssemblyRound
- PickPlace
- PickPlaceSingle
- PickPlaceMilk
- PickPlaceBread
- PickPlaceCereal
- PickPlaceCan
- Door
- Wipe
- TwoArmLift
- TwoArmPegInHole
- TwoArmHandover


### Available Controllers
- JOINT_VELOCITY
- JOINT_TORQUE
- JOINT_POSITION
- OSC_POSITION
- OSC_POSE
- IK_POSE

### Available Cameraview names
- Frontview
- Birdview
- Agentview
- Sideview
- Robot0_robotview
- Robot0_eye_in_hand
- Shouldercamera0


## VMAIL

### Install VMAIL
```
1. git clone https://github.com/rmrafailov/VMAIL.git
2. apt-get install ffmpeg
```

3. Download expert data for VMAIL

- To set up tensorflow GPU using pip on Linux system, follow the steps [here](https://www.tensorflow.org/install/pip#linux). Once installed, check for successful install of tensorflow GPU: 
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

1. Install Tianyu’s robosuite fork for VMAIL

```
git clone https://github.com/tianyudwang/robosuite.git
cd robosuite && pip install -e .
```

2. Install tacto_learn

```
git clone https://github.com/tianyudwang/tacto_learn.git
cd tacto_learn && git checkout robosuite && pip install -e .
```


### Install LAPAL
```
1. git clone https://github.com/tianyudwang/LAPAL.git
2. cd LAPAL/stable-baselines3/
3. pip3 install -r requirements.txt
4. python3 lapal/scripts/train_lapal.py configs/halfcheetah/LAPAL.yml
```


### Install SAC-AE
```
git clone https://github.com/denisyarats/pytorch_sac_ae.git
pip3 install -e .
```

### Download and Install MUJOCO
```
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
mv mujoco210/ .mujoco/
```

**Note:** Make sure to properly set the environment variable `MUJOCO_PY_MUJOCO_PATH` in the `~/.bashrc` file. Either set it to “” (nothing) or set it to a non-standard location. For example, see below:

For default location: 
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=""
```
For non-standard location:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jovyan/saqibcephsharedvol/ERLab/IRL_Project/.mujoco/mujoco210/bin/
export MUJOCO_PY_MUJOCO_PATH=/home/jovyan/saqibcephsharedvol/ERLab/IRL_Project/.mujoco/mujoco210/
```

If mujoco-related error such as `mujoco not found`, please check this file: `/root/saqibcephsharedvol2/ERLab/IRL_Project/rlenv/lib/python3.10/site-packages/mujoco_py/utils.py`


## Possible Errors

1. 
```
/opt/conda/lib/python3.10/site-packages/glfw/__init__.py:912: GLFWError: (65544) b\'X11: The DISPLAY environment variable is missing\'
warnings.warn(message, GLFWError)
```

**Solution:** Please check [here](https://github.com/denisyarats/dmc2gym/issues/4)


2. 
```
ValueError: Error: could not parse OBJ file \'/home/envs/rlenv/lib/python3.8/site packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_0.obj\'. Object name = robot0_link0_vis_0, id = 8, line = 45, column = -1
```

**Solution:** robosuite==1.4.0 installs latest MuJoCo version as a dependent library. But MuJoCo 2.3.4 gives the above error. Reinstall mujoco==2.3.2 after installing robosuite==1.4.0 and the problem is solved. 

3. Zero episode reward and batch reward during training and evaluation was due to setting “reward_shaping” parameter to False which uses sparse rewards. Instead use “reward_shaping=True” which uses dense rewards and this way, I get non-zero episode and batch rewards. Why is this the case when we use sparse rewards? Research later. 


4.
```
GLFW error (code %d): %s 65544 b\'X11: The DISPLAY environment variable is missing\' 
mujoco_py.cymj.GlfwError: Failed to initialize GLFW
```

**Solution:**
The error message indicates an issue with the GLFW library initialization in the MuJoCo Python bindings. The GLFW library is used for creating windows and handling input in graphical applications. The specific error message suggests that the "DISPLAY" environment variable is missing, which is **commonly associated with running graphical applications in a headless environment (such as a remote server without a graphical interface)**.

**Check Display Environment Variable:** The error message indicates that the "DISPLAY" environment variable is missing. This variable is used to specify the display server for graphical applications. If you're working in a headless environment, you might not have access to a display. Make sure you are running the code in a graphical environment, or consider using a different machine with a display.

**Virtual Display (Xvfb):** If you're working on a server without a display, you can use a virtual display using the Xvfb (X virtual framebuffer) utility. Xvfb allows you to run graphical applications without a physical display. You can start Xvfb and set the "DISPLAY" environment variable before running your script. Here's an example:

1. Install Xvfb (if not installed)
```
apt-get install xvfb
```
2. Start Xvfb on display:1
```
Xvfb :1 -screen 0 1024x768x24 &
```
3. Set the DISPLAY environment variable
```
export DISPLAY=:1
```
4. Run your Python script
```
python your_script.py
```

**Headless Execution:** If you don't require any graphical output or interaction, you might be able to run your script in a headless mode. You can achieve this by disabling the rendering and visual components in your MuJoCo environment. In your Python script, you can use the following code to disable rendering:

