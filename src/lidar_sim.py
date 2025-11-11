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
from utils.helpers import setup_simulation, get_pose, setup_foxglove_lidar_schema


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    cfg = load_config()

    # Simulation Loop Parameters
    fps = cfg['simulation']['fps']
    total_frames = int(cfg['simulation']['duration_sec'] * fps)
    speed_mps = cfg['simulation']['speed_mps']
    debug_viz = cfg['simulation'].get('debug_viz', False)

    lidar_radius = cfg['lidar']['range_m']
    channels = cfg['lidar']['channels']  
    vertical_fov_deg = cfg['lidar']['vertical_fov_deg']
    num_azim_bins = cfg['lidar'].get('horizontal_bins', 1024)

    angle_threshold_rad = np.deg2rad(cfg['lidar'].get('angle_threshold_deg', 0.5))

    ######### SETUP SIMULATION #########
    datapath, output_mcap, start_pos, R, motion_vec, K = setup_simulation(cfg, sensor_type='lidar')

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

    # build KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    vis = None 
    pcd_vis = o3d.geometry.PointCloud() # Placeholder
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0) # Sensor origin

    if debug_viz:
        print("Debug 3D visualization enabled. A window will open.")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        opt = vis.get_render_option()
        opt.point_size = 5.0
        opt.background_color = np.asarray([1, 1, 1])
        vis.add_geometry(origin_frame)        
        vis.add_geometry(pcd_vis)


    with open(output_mcap, "wb") as f:
        writer = Writer(f)
        writer.start()

        # register schema and channel for foxglove point cloud
        schema_id, channel_id = setup_foxglove_lidar_schema(writer)

        print(f"Starting simulation ({total_frames} frames)...")
        start_ns = time.time_ns()

        for i in range(total_frames):
            start_process = time.time()
            
            # get the current pose
            curr_pose = get_pose(i, total_frames, start_pos, motion_vec, speed_mps, fps)

            # lets make a bubble with KD-Tree for faster point selection for large point clouds
            [k, idx, _] = kdtree.search_radius_vector_3d(curr_pose, lidar_radius)
            if len(idx) == 0: continue

            print("Points reduced to :", len(idx))

            local_points = points_world[:, idx]
            local_colors = colors_world[:, idx]
            # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local_points.T))])

            #CORE LOGIC
            #convert local points relative to lidar frame
            points_lidar = local_points - curr_pose.reshape(3, 1)

            #based on no. of beams, filter
            elevation_arr = np.linspace(
                np.deg2rad(vertical_fov_deg[0]), 
                np.deg2rad(vertical_fov_deg[1]), 
                channels)
            
            range_buffer = np.full((channels, num_azim_bins), np.inf) 
            index_buffer = np.full((channels, num_azim_bins), -1, dtype=int)
            
            # print(elevation_arr)

            for j, point in enumerate(points_lidar.T):
                x,y,z = point
                
                radius = np.linalg.norm(point, axis=0) + 1e-9
                safe_ratio = np.clip(z / radius, -1.0, 1.0)
                elev = np.arcsin(safe_ratio)
                azim = np.arctan2(y, x)

                elev_idx = np.argmin(np.abs(elevation_arr - elev))
                nearest_elev_rad = elevation_arr[elev_idx]

                if abs(nearest_elev_rad - elev) <= angle_threshold_rad:
                    
                    # Convert to index (0 to num_azim_bins-1)
                    azim_idx = int(((azim + np.pi) / (2 * np.pi)) * num_azim_bins)
                    
                    # precaution
                    if azim_idx < 0 or azim_idx >= num_azim_bins or elev_idx < 0 or elev_idx >= channels:
                        continue

                    # occlusion check
                    if radius < range_buffer[elev_idx, azim_idx]:
                        range_buffer[elev_idx, azim_idx] = radius
                        index_buffer[elev_idx, azim_idx] = j

            # get unique points, filter out -1s
            final_indices = index_buffer[index_buffer != -1]

            final_points_lidar = points_lidar[:, final_indices] 
            final_colors_lidar = local_colors[:, final_indices]

            final_num_points = final_points_lidar.shape[1]

            if debug_viz:

                vis.remove_geometry(pcd_vis, reset_bounding_box=True) 
                pcd_vis = o3d.geometry.PointCloud()
                pcd_vis.points = o3d.utility.Vector3dVector(final_points_lidar.T)
                pcd_vis.colors = o3d.utility.Vector3dVector(final_colors_lidar.T)
                vis.add_geometry(pcd_vis, reset_bounding_box=True)
                
                if not vis.poll_events():
                    print("\nUser closed the visualization window.")
                    debug_viz = False
                
                vis.update_renderer()
            
            if final_num_points == 0:
                continue # skip frame if no points are visible

            end_process = time.time()
            print(f"Frame {i+1}/{total_frames} processed in {(end_process - start_process)*1000:.1f} ms ({final_num_points} points)")

    if vis:
        vis.destroy_window()
    

if __name__ == "__main__":
    main()