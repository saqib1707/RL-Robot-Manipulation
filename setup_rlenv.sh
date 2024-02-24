apt-get update
apt-get install -y vim curl wget unzip tar git tree net-tools htop tmux python3-dev python3-pip python3.8-venv
apt-get install -y libglfw3 libglew-dev libosmesa6-dev libgl1-mesa-dev libgl1-mesa-glx libegl1-mesa libopengl0 patchelf
apt-get install -y xvfb

# create virtual env
mkdir /home/envs
python3.8 -m venv /home/envs/rlenv
source /home/envs/rlenv/bin/activate
cd /home/saqibcephsharedvol2/ERLab/IRL_Project/
pip3 install -r requirements_rlenv.txt

cd dmc2gym/
pip3 install -e ./
cd ../
python3 /home/envs/rlenv/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py

# install modified stable-baselines3 from source
# cd LAPAL/stable-baselines3/
# pip3 install -e .

# install LAPAL
# cd ..
# pip3 install -e .

# install Tianyu's robosuite (for touch obs)
cd robosuite/
pip3 install -e ./
cd ../

deactivate
cp /home/saqibcephsharedvol2/ERLab/IRL_Project/nautilus_utils/.bashrc /root/.bashrc
cp /home/saqibcephsharedvol2/ERLab/IRL_Project/nautilus_utils/.bash_prompt /root/.bash_prompt
cp /home/saqibcephsharedvol2/ERLab/IRL_Project/nautilus_utils/.bash_aliases /root/.bash_aliases

# make this last step as it requires manual input
ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
dpkg-reconfigure -f noninteractive tzdata
timedatectl set-timezone America/Los_Angeles
apt-get install -y ffmpeg

source /root/.bashrc # (this does not work)