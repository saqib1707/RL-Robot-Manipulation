apt-get update
apt-get install -y vim curl wget unzip tar git net-tools htop tmux python3-dev python3-pip python3.8-venv
apt-get install -y libglfw3 libglew-dev libosmesa6-dev libgl1-mesa-dev libgl1-mesa-glx libegl1-mesa libopengl0 patchelf
mkdir /home/envs
python3.8 -m venv /home/envs/rlenv
source /home/envs/rlenv/bin/activate
cd /home/saqibcephsharedvol2/ERLab/IRL_Project/
pip3 install -r reqs_rlenv.txt
cd dmc2gym/
pip3 install -e ./
cd ../
apt-get install -y ffmpeg
python3 /home/envs/rlenv/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py
deactivate
cp nautilus_utils/.bashrc /root/.bashrc
# source ~/.bashrc # (this does not work)