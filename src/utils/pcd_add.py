import open3d as o3d
from pathlib import Path

curr_dir = Path.cwd()
ply_path = curr_dir.parent.parent / "dataset" / "Toronto_3D"
pcd_list = ["L001.ply", "L002.ply"]

pcd_combined = o3d.geometry.PointCloud()
for filename in pcd_list:
    pcd = o3d.io.read_point_cloud(str(ply_path / filename))
    pcd_combined += pcd

# pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.05)

o3d.io.write_point_cloud(str(ply_path / "Toronto_combined.ply"), pcd_combined)
o3d.visualization.draw_geometries([pcd_combined])


if __name__ == "__main__":
    pass
