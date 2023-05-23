# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# don't put duplicate lines in the history. See bash(1) for more options
# ... or force ignoredups and ignorespace
HISTCONTROL=ignoredups:ignorespace

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "$debian_chroot" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        # We have color support; assume it's compliant with Ecma-48
        # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
        # a case would tend to support setf rather than setaf.)
        color_prompt=yes
    else
        color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias c='clear'
alias activate='source /home/envs/rlenv/bin/activate'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

if [ -f ~/.bash_prompt ]; then
    . ~/.bash_prompt
fi

# sets up the bash prompt
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
PS1="\[\033[32m\]\u";
PS1+="@";
PS1+="\[\033[32m\]\h";
PS1+=" : ";
PS1+="\[\033[33;1m\]\w";
PS1+="\[\033[32m\]\$(parse_git_branch)";
PS1+="\n\[\033[m\]$ ";
# export PS1;

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
#if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
#    . /etc/bash_completion
#fi

# for the job
export MUJOCO_GL=osmesa     # (for headless linux without X-display)
# export MUJOCO_GL=glfw     # (on MacOS, use glfw backend instead)
# export MUJOCO_GL=egl

# below line for GPU rendering
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# below line only for a headless server
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/saqibcephsharedvol2/ERLab/IRL_Project/.mujoco/mujoco210/bin
# export MUJOCO_PY_MUJOCO_PATH=/home/saqibcephsharedvol2/ERLab/IRL_Project/.mujoco/mujoco210/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
# export MUJOCO_PY_MUJOCO_PATH=""

# export MJLIB_PATH=/home/saqibcephsharedvol2/ERLab/IRL_Project/.mujoco/mujoco210/bin/libmujoco210.dylib
# export MJKEY_PATH=/home/saqibcephsharedvol2/ERLab/IRL_Project/.mujoco/mujoco210/mjkey.txt
# export MUJOCO_PY_MJPRO_PATH=/home/saqibcephsharedvol2/ERLab/IRL_Project/.mujoco/mujoco210/
# export MUJOCO_PY_MJKEY_PATH=/home/saqibcephsharedvol2/ERLab/IRL_Project/.mujoco/mujoco210/mjkey.txt

# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
CUDNN_PATH='/home/envs/rlenv/lib/python3.8/site-packages/nvidia/cudnn'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib
