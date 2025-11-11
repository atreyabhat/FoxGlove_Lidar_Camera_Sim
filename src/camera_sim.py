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

    # Simulation Loop Parameters
    fps = cfg['simulation']['fps']
    total_frames = int(cfg['simulation']['duration_sec'] * fps)
    speed_mps = cfg['simulation']['speed_mps']
    W, H = cfg['camera']['width'], cfg['camera']['height']

    with open(output_mcap, "wb") as f:
        writer = Writer(f)
        writer.start()

        # register schema and channel for foxglove.CompressedImage
        schema_id = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding=SchemaEncoding.JSONSchema,
            data='{"type": "object", "properties": {"timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}}, "frame_id": {"type": "string"}, "format": {"type": "string"}, "data": {"type": "string", "contentEncoding": "base64"}}}'.encode('utf-8')
        )
        channel_id = writer.register_channel("camera_front", MessageEncoding.JSON, schema_id)

        print(f"Starting simulation ({total_frames} frames)...")
        start_ns = time.time_ns()

        for i in range(total_frames):
            print(f"Processing frame {i+1}/{total_frames}...", end="\r")
            
            t = get_pose(i, total_frames, start_pos, motion_vec, speed_mps, fps)

            # transform and project points
            points_cam = R @ (points_world - t.reshape(3, 1))

            valid_mask = points_cam[2, :] > 0.1
            p_valid = points_cam[:, valid_mask]
            c_valid = colors_world[:, valid_mask]

            if p_valid.shape[1] == 0: continue

            p_proj = K @ p_valid
            p_proj[2, p_proj[2, :] == 0] = 1e-6
            u = (p_proj[0, :] / p_proj[2, :]).astype(int)
            v = (p_proj[1, :] / p_proj[2, :]).astype(int)
            depths = p_proj[2, :]

            # filter with image bounds
            mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, depths = u[mask], v[mask], depths[mask]
            c_final = c_valid[:, mask]

            # render image
            img = np.zeros((H, W, 3), dtype=np.uint8)
            if u.size > 0:
                order = np.argsort(depths)[::-1]
                u_sorted, v_sorted = u[order], v[order]
                c_sorted = c_final[:, order]
                colors_bgr = (c_sorted.T[:, [2, 1, 0]] * 255).astype(np.uint8)
                img[v_sorted, u_sorted] = colors_bgr

            # post-processing: blur and noise
            if cfg['camera']['blur_enabled']:
                k = cfg['camera']['blur_kernel_size']
                img = cv2.GaussianBlur(img, (k, k), 0)
            if cfg['camera']['noise_enabled']:
                noise = np.random.normal(0, cfg['camera']['noise_sigma'], img.shape)
                img = cv2.add(img, noise.astype(np.uint8))

            # encode to JPEG and write to mcap
            _, jpeg_data = cv2.imencode('.jpg', img)
            frame_time_ns = start_ns + int(i * (1e9 / fps))
            msg = {
                "timestamp": {"sec": frame_time_ns // 10**9, "nsec": frame_time_ns % 10**9},
                "frame_id": "camera_link",
                "format": "jpeg",
                "data": base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
            }
            writer.add_message(channel_id, frame_time_ns, json.dumps(msg).encode('utf-8'), frame_time_ns)

    print(f"\nCamera simulation finished! Output: {output_mcap}")

if __name__ == "__main__":
    main()