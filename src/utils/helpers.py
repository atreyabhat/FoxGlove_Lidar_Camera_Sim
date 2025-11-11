import numpy as np
from pathlib import Path

def setup_simulation(cfg):
    """Prepares all static simulation data."""
    base_dir = Path.cwd()
    datapath = str(base_dir / cfg["paths"]["dataset_dir"] / cfg["paths"]["ply_filename"])
    output_path = str(base_dir / cfg["paths"]["result_dir"] / cfg["paths"]["camera_mcap_filename"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # World Setup
    start_pos_world = np.array(cfg["world"]["start_pos"])
    utm_offset = np.array(cfg["world"]["utm_offset"])

    # Camera's start position relative to the shifted origin
    start_pos_relative = start_pos_world - utm_offset

    # Camera Rotation
    # Default R
    R_pitch = np.array([[1, 0, 0], 
                        [0, 0, -1], 
                        [0, 1, 0]])

    # Adding small Yaw (trial and error correction)
    theta = np.radians(cfg["camera"]["yaw_offset_deg"]) + np.pi
    c, s = np.cos(theta), np.sin(theta)
    R_yaw = np.array([[c, -s, 0], 
                      [s, c, 0], 
                      [0, 0, 1]])

    # final rotation
    R_world_to_cam = R_pitch @ R_yaw

    # motion Vector
    R_cam_to_world = R_world_to_cam.T
    motion_dir = R_cam_to_world[:, 2]  # forward direction in world frame

    #fix drift
    motion_dir[2] = motion_dir[2] - 0.02  # eliminating vertical drift, need a better solution

    motion_vec = motion_dir / np.linalg.norm(motion_dir)  # normalize

    # camera intrinsics
    W, H = cfg["camera"]["width"], cfg["camera"]["height"]
    FL = W / 2.0
    K = np.array([[FL, 0, W / 2.0], 
                  [0, FL, H / 2.0], 
                  [0, 0, 1]])

    return datapath, output_path, start_pos_relative, R_world_to_cam, motion_vec, K



def get_pose(frame_idx, total_frames, start_pos, motion_vec, speed_mps, fps):
    """Calculates the camera position for a given frame based on speed."""
    time_elapsed = frame_idx / fps  # time elapsed
    distance = speed_mps * time_elapsed  # distance traveled
    current_pos = start_pos + (motion_vec * distance)  # curr position

    return current_pos
