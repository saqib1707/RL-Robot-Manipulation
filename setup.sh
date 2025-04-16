#!/bin/bash

# install necessary packages
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS using Homebrew
    brew update
    brew install vim curl wget unzip tar git tree net-tools htop tmux python3 python@3.8 glfw glew mesa patchelf
elif [[ "$(uname)" == "Linux" ]]; then
    # Linux using APT
    apt update
    apt install -y vim curl wget unzip tar git tree net-tools htop tmux python3-dev python3-pip python3-venv \
                        libglfw3 libglew-dev libosmesa6-dev libgl1-mesa-dev libgl1-mesa-glx libegl1-mesa libopengl0 patchelf \
                        xvfb
else
    echo "Unsupported operating system"
    exit 1
fi

# create virtual environment
ENV_NAME="rlenv"
PROJECT_DIR="ERLab/robot_manipulation/"
if [[ "$(uname)" == "Darwin" ]]; then
    HOMEDIR="$HOME"
elif [[ "$(uname)" == "Linux" ]]; then
    HOMEDIR="/home/"
fi

mkdir -p "$HOMEDIR/.venvs/"
python3 -m venv "$HOMEDIR/.venvs/$ENV_NAME"
source "$HOMEDIR/.venvs/$ENV_NAME/bin/activate"

if [[ "$(uname)" == "Darwin" ]]; then
    cd "$HOMEDIR/Desktop/$PROJECT_DIR"
elif [[ "$(uname)" == "Linux" ]]; then
    cd "$HOMEDIR/saqibcephsharedvol2/$PROJECT_DIR"
fi
pip3 install -r requirements.txt

cd dmc2gym/ && pip3 install -e ./ && cd ..
python3 "$HOMEDIR/.venvs/$ENV_NAME/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py"

# install modified stable-baselines3 from source
# cd LAPAL/stable-baselines3/ && pip3 install -e .

# install LAPAL
# cd .. && pip3 install -e .

# install Tianyu's robosuite (for touch obs)
cd robosuite/ && pip3 install -e ./ && cd ..
deactivate

# copy config files
if [[ "$(uname)" == "Linux" ]]; then
    cp "$HOMEDIR/saqibcephsharedvol2/$PROJECT_DIR/nautilus_utils/.bashrc" "$HOME/.bashrc"
    cp "$HOMEDIR/saqibcephsharedvol2/$PROJECT_DIR/nautilus_utils/.bash_prompt" "$HOME/.bash_prompt"
    cp "$HOMEDIR/saqibcephsharedvol2/$PROJECT_DIR/nautilus_utils/.bash_aliases" "$HOME/.bash_aliases"
fi

# Set timezone (# make this last step as it requires manual input)
if [[ "$(uname)" == "Linux" ]]; then
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
    dpkg-reconfigure -f noninteractive tzdata
    timedatectl set-timezone America/Los_Angeles
    apt install -y ffmpeg
fi

# Source the bashrc (not applicable for macOS)
if [[ "$(uname)" == "Linux" ]]; then
    source $HOME/.bashrc    # (this does not work)
fi