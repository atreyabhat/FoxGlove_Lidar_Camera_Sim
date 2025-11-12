"""
This script simulates a LiDAR model moving through a static pointcloud. 
"""

import numpy as np
import open3d as o3d
import time
import yaml
from pathlib import Path
import foxglove
from foxglove.channels import PointCloudChannel, SceneUpdateChannel
from foxglove.schemas import (
    Timestamp, PointCloud, PackedElementField, 
    PackedElementFieldNumericType, Pose, Vector3, Quaternion
)
from foxglove.schemas import SceneUpdate, SceneEntity, ModelPrimitive
from utils.helpers import setup_simulation, get_pose
from scipy.spatial.transform import Rotation as R

f32 = PackedElementFieldNumericType.Float32



def rotmat_to_quat(rot_matrix):
    quat = R.from_matrix(rot_matrix).as_quat()
    return dict(x=quat[0], y=quat[1], z=quat[2], w=quat[3])


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def main():
    cfg = load_config()

    # Simulation Loop Parameters
    fps = cfg['simulation']['fps']
    total_frames = int(cfg['simulation']['duration_sec'] * fps)
    speed_mps = cfg['simulation']['speed_mps']

    lidar_radius = cfg['lidar']['range_m']
    channels = cfg['lidar']['channels']  
    vertical_fov_deg = cfg['lidar']['vertical_fov_deg']
    num_azim_bins = cfg['lidar'].get('horizontal_bins', 1024)

    angle_threshold_rad = np.deg2rad(cfg['lidar'].get('angle_threshold_deg', 0.5))

    ######### SETUP SIMULATION #########
    datapath, output_mcap, start_pos, R, R_yaw, motion_vec, K = setup_simulation(cfg, sensor_type='lidar')


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

    # per frame precomputations
    elevation_arr = np.linspace(
        np.deg2rad(vertical_fov_deg[0]), 
        np.deg2rad(vertical_fov_deg[1]), 
        channels
    )

    elev_min_rad = elevation_arr[0]
    elev_span_rad = elevation_arr[-1] - elev_min_rad
    horiz_fov_rad = 2 * np.pi
    
    
    with foxglove.open_mcap(output_mcap, allow_overwrite=True) as writer:

        lidar_topic = cfg['lidar']['topic']
        channel = PointCloudChannel(topic=lidar_topic)

        fields = [
            PackedElementField(name="x", offset=0, type=f32),
            PackedElementField(name="y", offset=4, type=f32),
            PackedElementField(name="z", offset=8, type=f32),
            PackedElementField(name="r", offset=12, type=f32),
            PackedElementField(name="g", offset=16, type=f32),
            PackedElementField(name="b", offset=20, type=f32)
        ]
        point_stride = 24

        print(f"Starting simulation ({total_frames} frames)...")
        start_ns = 1704067200 * 1_000_000_000

        print("Adding car model...")
        car_channel = SceneUpdateChannel(topic="/car_model")

        model_path = (Path(__file__).parent.parent / "assets"/ "f1_car_concept.glb").as_uri()
        
        
        car_scene = SceneUpdate(
            entities=[
                SceneEntity(
                    timestamp=Timestamp(sec=0, nsec=0), 
                    frame_id="base_link",
                    id="the_car",
                    models=[
                        ModelPrimitive(
                            url=model_path,
                            scale=Vector3(x=0.6, y=0.6, z=0.6),
                            pose=Pose(
                                position=Vector3(x=0, y=0, z=-1.0),
                                orientation=Quaternion(x=0, y=0, z=0.707, w=0.707)
                            )
                        )
                    ]
                )
            ]
        )

        car_channel.log(car_scene, log_time=start_ns)

        for i in range(total_frames):
            start_process = time.time()
            
            # get the current pose
            curr_pose = get_pose(i, total_frames, start_pos, motion_vec, speed_mps, fps)

            # lets make a bubble with KD-Tree for faster point selection for large point clouds
            [k, idx, _] = kdtree.search_radius_vector_3d(curr_pose, lidar_radius)
            if len(idx) == 0: continue

            local_points = points_world[:, idx]
            local_colors = colors_world[:, idx]

            #convert local points relative to lidar frame
            points_lidar = local_points - curr_pose.reshape(3, 1)

            # optimization
            num_pts_in_radius = points_lidar.shape[1]
            if num_pts_in_radius == 0:
                continue
                
            # vectorixing
            x,y,z = points_lidar[0,:], points_lidar[1,:], points_lidar[2,:]
            radius = np.linalg.norm(points_lidar, axis=0) + 1e-9
            elev = np.arcsin(np.clip(z / radius, -1.0, 1.0))
            azim = np.arctan2(y, x)
            
            #scaling
            elev_normalized = (elev - elev_min_rad) / elev_span_rad
            elev_idx = np.round(elev_normalized * (channels - 1)).astype(int)
            
            azim_normalized = (azim + np.pi) / horiz_fov_rad
            azim_idx = np.round(azim_normalized * num_azim_bins).astype(int)

            # print(azim_idx.shape)
            # print(elev_idx.shape)

            #check bounds for horiz bins and verticle channels
            # from here on, masks propagate in the pipeline leading to a combined mask
            in_bounds_mask = (elev_idx >= 0) & (elev_idx < channels) & (azim_idx >= 0) & (azim_idx < num_azim_bins)
            
            # create angle mask that filters all points in elev channels with angle threshold
            nearest_elev_rad = elevation_arr[elev_idx[in_bounds_mask]]
            angle_mask = np.abs(nearest_elev_rad - elev[in_bounds_mask]) <= angle_threshold_rad
            
            # get the final combined mask
            final_mask = np.zeros(num_pts_in_radius, dtype=bool)
            final_mask[in_bounds_mask] = angle_mask

            elev_idx_f = elev_idx[final_mask]
            azim_idx_f = azim_idx[final_mask]
            radius_f = radius[final_mask]

            #track original indices for colors
            original_indices = np.arange(num_pts_in_radius)[final_mask]
            
            # Sort all points by radius (closest first)
            sort_order = np.argsort(radius_f)
            elev_idx_s = elev_idx_f[sort_order]
            azim_idx_s = azim_idx_f[sort_order]
            original_indices_s = original_indices[sort_order]
            
            # ravel 2D beam idx to 1D 
            raveled_indices = np.ravel_multi_index((elev_idx_s, azim_idx_s), (channels, num_azim_bins))
            _, unique_map_indices = np.unique(raveled_indices, return_index=True)
            final_indices = original_indices_s[unique_map_indices]


            final_points_lidar = points_lidar[:, final_indices] 
            final_colors_lidar = local_colors[:, final_indices]
            final_num_points = final_points_lidar.shape[1]
            
            
            if final_num_points == 0:
                continue #skip

            
            points_xyz = final_points_lidar.T.astype(np.float32)
            points_rgb = final_colors_lidar.T.astype(np.float32)

            combined_data = np.hstack((points_xyz, points_rgb))
            data_bytes = combined_data.tobytes()

            log_time_ns = start_ns + int((i / fps) * 1e9)
            ts_obj = Timestamp(sec=log_time_ns // int(1e9), nsec=log_time_ns % int(1e9))

            q_dict = rotmat_to_quat(R_yaw)
            
            pose_obj = Pose(
                position=Vector3(x=0.0, y=0.0, z=0.0),
                orientation=Quaternion(x=q_dict['x'], y=q_dict['y'], z=q_dict['z'], w=q_dict['w'])
            )
            
            pc_msg = PointCloud(
                timestamp=ts_obj,
                frame_id="base_link",
                pose=pose_obj,        
                point_stride=point_stride,
                fields=fields,
                data=data_bytes
            )

            channel.log(pc_msg, log_time=log_time_ns)

            end_process = time.time()
            print(f"Frame {i+1}/{total_frames} processed in {(end_process - start_process)*1000:.1f} ms", end=" " * 10 + "\r")


if __name__ == "__main__":
    main()