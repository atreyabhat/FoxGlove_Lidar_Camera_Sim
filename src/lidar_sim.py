import numpy as np
import open3d as o3d
import cv2
import time
import yaml
import json
import base64
from pathlib import Path
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
from utils.helpers import setup_simulation, get_pose

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    cfg = load_config()
    datapath, output_mcap, start_pos, R, motion_vec, K = setup_simulation(cfg)

    print(f"Loading point cloud from {datapath}...")
    pcd = o3d.io.read_point_cloud(datapath)
    
    #downsample if enabled
    if cfg['processing']['downsample_enabled']:
        voxel_size = cfg['processing']['voxel_size_m']
        print(f"Downsampling with voxel size {voxel_size}m...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # offset points to UTM origin as described by the authors
    points = np.asarray(pcd.points) - np.array(cfg['world']['utm_offset'])
    pcd.points = o3d.utility.Vector3dVector(points)

    points_world = np.asarray(pcd.points).T
    colors_world = np.asarray(pcd.colors).T
    print(f"Loaded {points_world.shape[1]} points.")

    

if __name__ == "__main__":
    main()