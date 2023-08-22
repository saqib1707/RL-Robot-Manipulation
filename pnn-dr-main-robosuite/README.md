# PNN-DR

Virtual column implementation based on the architecture proposed by Rusu et al. (2017) in "Sim-to-Real Robot Learning from Pixels with Progressive Nets" plus Domain Randomization to improve the agent's performance when some environment features change.

## Description

Regarding the efficiency issue on transfering the knowledge between a virtual environment and the real scenario (sim-to-real) in deep reinforcement learning problems, we have implemented the PNN-virtual column proposed by Rusu et al. (2017) and enhanced its performance with the usage of Domain Randomization for the camera pose.

The problem description, MDP design, results obatained and comparison between the models can be checkup in the manuscript that has been send to Applied Intellignece (Springer), **currently under review**, "Learning more with the same effort: how randomization improves the robustness of a robotic deep reinforcement learning agent" by L. Güitta-López, J. Boal and Á. J. López-López (2022).

## Files structure

The project structure is the following:
```
.
├── irb120
│   ├── assets
│   │   ├── irb120.xml
│   │   └── stl
│   │       ├── base.stl
│   │       ├── connection_part.stl
│   │       ├── link1.stl
│   │       ├── link2.stl
│   │       ├── link3.stl
│   │       ├── link4.stl
│   │       ├── link5.stl
│   │       ├── link6.stl
│   │       ├── gripper_right.stl
│   │       ├── gripper_left.stl
│   │       └── schunk.stl
│   ├── __init__.py
│   └── irb120.py
├── main_irb120.py
├── model.py
├── optim.py
├── requirements.txt
├── test.py
├── train.py
└── utils.py
```

Notes:

* If you want to change the robot model, you should replace the irb120.xml file inside the assests folder and the part in the irb120.py file that instantiates that model.

# Installation

The environment is developped as an Open AI Gym env and the physics simulator chosen is MuJoCo. The robotic arm is an IRB120 from ABB.

## OS
The project was tested under Ubuntu 20.04 with Python 3.8. The use of a GPU is not mandatory but desirable to speed up the training and test.

## Setup
* **MuJoCo**

Please follow the instructions available at https://github.com/deepmind/mujoco to download version '2.1.0'.

* **mujoco-py**

Please follow the instructions available at https://github.com/openai/mujoco-py to download version '2.1.2.14'.

* **Torch**

Please follow the instructions available at https://pytorch.org/get-started/locally/ to download torch according to your software settings.

* **Other packages**

For the remaining packages, please install the requirements.txt file.
```
pip3 install -r requirements.txt
```

# Getting started

## Training
To train the model run in a terminal:
```
python3 main_irb120.py --overwrite --fine_render --reward_continuous --num-processes 17
```

Note that the number of processes will depend on your computer resources, so you can set it at your convenience.

## Evaluation

To evaluate a model, place the .pth at the same lavel as the main file an run:
```
python3 main_irb120.py --overwrite --fine_render --reward_continuous --evaluate --model 45752843_model.pth
```

Note that the name of the model might vary depending on the results achieved. 

The other arguments that might be customize are explained in the main file.

## License

MIT license.
