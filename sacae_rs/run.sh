cd /home/saqibcephsharedvol2/ERLab/IRL_Project/sacae_rs/
python3 train_suite.py \
--controller="" \
--save_tb \
--save_video \
--image_size=84 \
--num_layers=4 \
--frame_stack=3 \
--action_repeat=1 \
--horizon=400 \
--init_steps=1000 \
--eval_freq=10000 \
--num_eval_episodes=2 \
--video_save_freq=50000 \
--work_dir='./logdir/' \
