# import sys
# sys.path.append("/home/saqibcephsharedvol2/ERLab/IRL_Project/SceneGrasp/")

# from common.utils.nocs_utils import load_depth
# from common.utils.misc_utils import (
#     convert_realsense_rgb_depth_to_o3d_pcl,
#     get_o3d_pcd_from_np,
#     get_scene_grasp_model_params,
# )
# from common.utils.scene_grasp_utils import (
#     SceneGraspModel,
#     get_final_grasps_from_predictions_np,
#     get_grasp_vis,
# )

from datetime import datetime, timezone
import pytz

us_pacific_dt = datetime.now(pytz.timezone('US/Pacific'))
print(us_pacific_dt)


print(datetime.now().strftime('%Y%m%dT%H%M%S'))