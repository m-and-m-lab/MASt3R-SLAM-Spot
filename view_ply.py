import open3d as o3d

# Replace with your PLY file path
ply_path = "logs/sample.ply"

# Load the point cloud
pcd = o3d.io.read_point_cloud(ply_path)

# Print basic info
print(pcd)
print("Number of points:", len(pcd.points))

# Visualize
o3d.visualization.draw_geometries([pcd])