import numpy as np
import open3d as o3d

CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ])

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [
        0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)


def create_camera_actor(g, scale=0.05):
    """build open3d camera polydata"""
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def get_lineset(points, lines, color=[1, 0, 0]):
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_trajectory(slam):
    trajectory = slam.get_raw_trajectory()
    trajectory = np.array(trajectory)
    lines = [[p, p + 1] for p in range(trajectory.shape[0] - 1)]
    trajectory = get_lineset(trajectory, lines)
    return trajectory


def get_keyframe_graph(slam):
    keyframes = slam.get_keyframe_poses()
    camera = []
    for f in keyframes:
        cam = create_camera_actor(0.3)
        cam.transform(f)
        camera += [cam]
    return camera
