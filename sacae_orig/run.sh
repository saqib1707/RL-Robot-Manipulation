cd /home/saqibcephsharedvol2/ERLab/IRL_Project/sacae_orig/
python3 train.py \
--save_tb \
--save_video \
--image_size=84 \
--num_layers=4 \
--action_repeat=4 \
--init_steps=1000 \
--eval_freq=1000 \
--num_eval_episodes=2 \
--video_save_freq=1000 \
--work_dir='./logtemp/' \
