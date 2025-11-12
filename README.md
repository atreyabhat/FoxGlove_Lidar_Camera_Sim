## Simulating Camera and Lidar sensors moving through a static pointcloud and visualizing with FoxGlove


The package replays static datasets (e.g., Toronto_3D) through a virtual ego-vehicle, emits MCAP logs that follow Foxglove schemas, and makes it easy to visualize synchronized RGB and point cloud streams.

## Features
- Camera renderer that projects colorized point clouds with distortion modeling and JPEG-compressed MCAP output.
- Simple-Lama image inpainting is implemented for the camera render, producing much more realistic images
  
- LiDAR simulator that enforces per-channel vertical FOV, horizontal binning, occlusion, and logs to Foxglove `PointCloud` messages.
- Configurable trajectory, frame rate, downsampling, and dataset paths via a single `config.yaml`.
- Visualize results in `results/*.mcap`, compatible with Foxglove.


## Quick Start
1. `python -m venv venv && source venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`.
3. Place/confirm your `.ply` map under `dataset/` and update paths in `cam_lidar_sim_foxglove/src/config.yaml`.
4. Run the simulators:
   ```bash
   cd cam_lidar_sim_foxglove/src
   python camera_sim_optimized.py
   python lidar_sim_optimized.py
   ```
5. Open the generated MCAP files from `results/` in Foxglove to inspect the data streams.

<img width="1383" height="1107" alt="Screenshot 2025-11-12 at 3 44 11 PM" src="https://github.com/user-attachments/assets/5c0ed64b-a24c-4862-a1de-3538fdb9e8c1" />



## Repository Layout
- `cam_lidar_sim_foxglove/src/`: Simulator scripts, configuration, and shared helpers.
- `dataset/`: Expected location for input point clouds (Toronto_3D sample paths provided).
- `results/`: Destination for MCAP logs.
- `assets/`:  Reference assets.

Tweak `config.yaml` to experiment with different sensor placements, motion profiles, or processing parameters, and rerun the scripts to regenerate fresh logs.
