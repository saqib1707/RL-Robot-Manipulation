cd /home/saqibcephsharedvol2/ERLab/IRL_Project/VMAIL/robosuite_task/

# python3 vmail.py \
# --expert_datadir="../expert_data/robosuite_expert/Lift/Panda/OSC_POSE/20230820T115652-robot0_eye_in_hand-250/expert_data/" \
# --camera_names='agentview' \
# --time_limit=250 \
# --horizon=15 \
# --task='PickPlaceBread' \
# --use_camera_obs=True \
# --use_object_obs=False \
# --use_depth_obs=False \
# --use_proprio_obs=False \
# --use_tactile_obs=False \
# --use_touch_obs=False \
# --store=True \
# --batch_size=32

python3 vmail.py \
--expert_datadir="../expert_data/robosuite_expert/PickPlaceBread/Panda/OSC_POSE/20230708T191838-agentview-250/expert_data/" \
--camera_names='agentview' \
--time_limit=250 \
--horizon=15 \
--task='PickPlaceBread' \
--use_camera_obs=True \
--use_proprio_obs=False \
--use_depth_obs=False \
--use_shape_obs=False \
--store=True


# python3 vmail.py \
# --expert_datadir="../expert_data/robosuite_expert/PickPlaceBread/Panda/OSC_POSE/20230708T195625-frontview-250/expert_data/" \
# --camera_names='frontview' \
# --time_limit=250 \
# --horizon=15 \
# --task='robosuite_PickPlaceBread_task' \
# --use_proprio_obs=False \
# --use_depth_obs=False \
# --store=True