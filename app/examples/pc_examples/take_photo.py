import open3d as o3d
import numpy as np
import os
from pathlib import Path

for folder_name in os.listdir("."):
    if folder_name != "take_photo.py":
        # Load point cloud
        pcd = o3d.io.read_point_cloud(Path(folder_name) / "pc.ply")

        # Set black points
        pcd.paint_uniform_color([0, 0, 0])  

        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=800)  # Keep same size for all images
        vis.add_geometry(pcd)

        # Set transparent background
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # White background (no transparency)


        # Capture Image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(Path(folder_name) / "pc.png", do_render=True)

        vis.destroy_window()
