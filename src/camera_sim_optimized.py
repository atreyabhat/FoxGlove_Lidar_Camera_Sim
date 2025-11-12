"""
This script simulates a pinhole camera moving through a static LIDAR pointcloud data. 
Once the image is rendered, LAMA Inpainting model is used to reconstruct the missing pieces in the render.
Finally both the images+timestamps are written to MCAP file in 2 topics.
"""

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
import numba
import torch
from simple_lama_inpainting import SimpleLama



#numba render for faster z-buffering
@numba.njit(fastmath=True)
def render_frame_numba(H, W, u, v, depths, colors_bgr):
    """
    Renders an image frame using a fast, JIT-compiled Z-buffer.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8) 
    #init all points to inf depth
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)

    for i in range(u.shape[0]):
        u_i, v_i = u[i], v[i]
        d = depths[i]

        if d < depth_buffer[v_i, u_i]:
            depth_buffer[v_i, u_i] = d
            img[v_i, u_i, :] = colors_bgr[i, :]

    return img


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    cfg = load_config()
    datapath, output_mcap, start_pos, R, R_yaw, motion_vec, K = setup_simulation(cfg, sensor_type='camera')

    dist_cfg = cfg['camera'].get('distortion', {})
    dist_enabled = dist_cfg.get('enabled', False)
    D = np.array(dist_cfg.get('D', [0.0, 0.0, 0.0, 0.0, 0.0])) 

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

    # Simulation Loop Parameters
    fps = cfg['simulation']['fps']
    total_frames = int(cfg['simulation']['duration_sec'] * fps)
    speed_mps = cfg['simulation']['speed_mps']
    W, H = cfg['camera']['width'], cfg['camera']['height']

    rvec_zeros = np.zeros(3, dtype=np.float32)
    tvec_zeros = np.zeros(3, dtype=np.float32)

    with open(output_mcap, "wb") as f:
        writer = Writer(f)
        writer.start()

        # register schema and channel for foxglove.CompressedImage
        schema_id = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding=SchemaEncoding.JSONSchema,
            data='{"type": "object", "properties": {"timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}}, "frame_id": {"type": "string"}, "format": {"type": "string"}, "data": {"type": "string", "contentEncoding": "base64"}}}'.encode('utf-8')
        )
        
        original_topic = cfg['camera']['topic']
        original_channel_id = writer.register_channel(original_topic, MessageEncoding.JSON, schema_id)
        inpainted_topic = original_topic + "/inpainted"
        inpainted_channel_id = writer.register_channel(inpainted_topic, MessageEncoding.JSON, schema_id)

        #load simple-lama
        device = "mps" if torch.mps.is_available() else "cpu"
        lama = SimpleLama(device=device)

        print("Lama model loaded.")
        print(f"Starting simulation ({total_frames} frames)...")

        start_ns = 1704067200 * 1_000_000_000 #hardcoded to sync lidar and camera

        for i in range(total_frames):
            start_process = time.time()
            
            curr_pose = get_pose(i, total_frames, start_pos, motion_vec, speed_mps, fps)

            # transform and project points
            points_cam = R @ (points_world - curr_pose.reshape(3, 1))
            valid_mask = points_cam[2, :] > 0.1
            p_valid = points_cam[:, valid_mask]
            c_valid = colors_world[:, valid_mask]

            if p_valid.shape[1] == 0: continue
            depths = p_valid[2, :]

            
            p_valid_cv = p_valid.T  #for opwncv 
            projected_pixels, _ = cv2.projectPoints(p_valid_cv,rvec_zeros, tvec_zeros, K, D)  #distortion coefficients
            projected_pixels = projected_pixels.squeeze()
            u = projected_pixels[:, 0].astype(int)
            v = projected_pixels[:, 1].astype(int)
            
            # filter with image bounds
            mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, depths = u[mask], v[mask], depths[mask]
            c_final = c_valid[:, mask]
            colors_bgr = (c_final.T[:, [2, 1, 0]] * 255).astype(np.uint8) 
            
            img = render_frame_numba(H, W, u, v, depths, colors_bgr)

            frame_time_ns = start_ns + int(i * (1e9 / fps))
            
            #publishing original sparse image
            # encode to JPEG and write to mcap
            _, jpeg_data = cv2.imencode('.jpg', img)
            msg = {
                "timestamp": {"sec": frame_time_ns // 10**9, "nsec": frame_time_ns % 10**9},
                "frame_id": "base_link",
                "format": "jpeg",
                "data": base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
            }
            writer.add_message(original_channel_id, frame_time_ns, json.dumps(msg).encode('utf-8'), frame_time_ns)


            # generate and publish the inpainted image
            mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))

            # didnt make much difference, commenting in the name of efficiency
            # kernel = np.ones((3,3), np.uint8)
            # mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            # cv2.imshow("Mask", mask_dilated)
            # cv2.waitKey(0)


            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_inpainted_pil = lama(img_rgb, mask)

            img_inpainted_rgb = np.array(img_inpainted_pil)
            img_inpainted_bgr = cv2.cvtColor(img_inpainted_rgb, cv2.COLOR_RGB2BGR)

            _, jpeg_data_inpainted = cv2.imencode('.jpg', img_inpainted_bgr)

            inpainted_msg = {
                "timestamp": {"sec": frame_time_ns // 10**9, "nsec": frame_time_ns % 10**9},
                "frame_id": "base_link",
                "format": "jpeg",
                "data": base64.b64encode(jpeg_data_inpainted.tobytes()).decode('utf-8')
            }
            writer.add_message(inpainted_channel_id, frame_time_ns, json.dumps(inpainted_msg).encode('utf-8'), frame_time_ns)
            
            end_process = time.time()
            print(f"Frame {i+1}/{total_frames} processed in {(end_process - start_process)*1000:.1f} ms", end=" " * 10 + "\r")

    print(f"\nCamera simulation finished! Output: {output_mcap}")


if __name__ == "__main__":
    main()