cd /home/saqibcephsharedvol2/ERLab/IRL_Project/VMAIL/robosuite_task/

# python3 vmail.py \
# --expert_datadir="../expert_data/robosuite_expert/Lift/Panda/OSC_POSE/20230628T100110-robot0_eye_in_hand-100/expert_data/" \
# --camera_names='robot0_eye_in_hand' \
# --time_limit=100 \
# --horizon=15

python3 vmail.py \
--expert_datadir="../expert_data/robosuite_expert/PickPlaceBread/Panda/OSC_POSE/20230708T191838-agentview-250/expert_data/" \
--camera_names='agentview' \
--time_limit=250 \
--horizon=15 \
--task='robosuite_PickPlaceBread_task' \
--use_object_obs=False \
--use_depth_obs=False \
--use_touch_obs=False \
--use_tactile_obs=False \
--store=True


# python3 vmail.py \
# --expert_datadir="../expert_data/robosuite_expert/PickPlaceBread/Panda/OSC_POSE/20230708T195625-frontview-250/expert_data/" \
# --camera_names='frontview' \
# --time_limit=250 \
# --horizon=15 \
# --task='robosuite_PickPlaceBread_task' \
# --use_object_obs=True \
# --use_depth_obs=True \
# --use_touch_obs=True \
# --use_tactile_obs=False \
# --store=True