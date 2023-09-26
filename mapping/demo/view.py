import argparse
import os
import pickle
from glob import glob
from time import sleep, time

import natsort
import numpy as np
import open3d as o3d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder, reset_bounding_box=False)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

    def merge_cylinder_segments(self):

        vertices_list = [np.asarray(mesh.vertices)
                         for mesh in self.cylinder_segments]
        triangles_list = [np.asarray(mesh.triangles)
                          for mesh in self.cylinder_segments]
        triangles_offset = np.cumsum([v.shape[0] for v in vertices_list])
        triangles_offset = np.insert(triangles_offset, 0, 0)[:-1]

        vertices = np.vstack(vertices_list)
        triangles = np.vstack(
            [triangle + offset for triangle, offset in zip(triangles_list, triangles_offset)])

        merged_mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(vertices),
                                                o3d.open3d.utility.Vector3iVector(triangles))
        color = self.colors if self.colors.ndim == 1 else self.colors[0]
        merged_mesh.paint_uniform_color(color)
        self.cylinder_segments = [merged_mesh]


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


def create_camera_actor2(g, pose, scale=0.05):
    """build open3d camera polydata"""
    points = o3d.utility.Vector3dVector(scale * CAM_POINTS)
    points = points @ pose[:3, :3].transpose() + pose[:3, 3]
    lines = o3d.utility.Vector2iVector(CAM_LINES)
    camera_actor = LineMesh(points, lines, [1, 0, 0], radius=0.005)
    camera_actor.merge_cylinder_segments()
    # color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    # camera_actor.paint_uniform_color(color)
    return camera_actor


def load_voxels(voxel_centre, voxel_size):
    voxels = []
    for i in range(len(voxel_centre)):
        voxel = voxel_centre[i]
        half_vsize = voxel_size / 2
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            voxel - half_vsize, voxel + half_vsize)
        bbox.color = [0, 0, 0]
        voxels += [bbox]
    return voxels


def load_mesh(mesh_data):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_data["verts"])
    mesh.triangles = o3d.utility.Vector3iVector(mesh_data["faces"])
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_data["color"])
    mesh.compute_vertex_normals()
    return mesh


def get_trajectory(poses, c=[1, 0, 0]):
    points = []
    for p in poses:
        points += [p[:3, 3]]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    # colors = [c for i in range(len(points))]
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(points)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.paint_uniform_color(c)
    line_set = LineMesh(points, lines, radius=0.005)
    line_set.merge_cylinder_segments()
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def load_data(scene_path):
    print(scene_path)
    data = pickle.load(open(scene_path, 'rb'))
    mesh = load_mesh(data["mesh"])
    voxel_size = data["voxel_size"]
    voxels = load_voxels(data["voxels"], voxel_size)
    pose = data["pose"]
    updated_poses = data["updated_poses"]
    keyframes = data["keyframes"]
    return mesh, voxels, pose, updated_poses, keyframes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_path", type=str)
    parser.add_argument("--frame_rate", type=int, default=10)
    args = parser.parse_args()

    scene_data = natsort.natsorted(
        glob(os.path.join(args.scene_path, "misc/scene_data_*.pkl")))
    mesh = None
    frame_rate = args.frame_rate
    frame_count = 0
    max_frame = len(scene_data)
    frame_poses = []
    play_start = False


    def update_mesh_and_voxels(vis):
        global mesh, frame_count, max_frame, frame_poses, scene_data
        if frame_count >= max_frame:
            return
        mesh, voxels, pose, updated_poses, keyframes = load_data(
            scene_data[frame_count])

        frame_poses += [pose]
        vis.clear_geometries()
        vis.add_geometry(mesh, reset_bounding_box=(frame_count == 0))
        for i in range(len(voxels)):
            vis.add_geometry(voxels[i], reset_bounding_box=False)
        # if len(frame_poses) > 1:
        #     trajectory = get_trajectory(frame_poses)
        #     vis.add_geometry(trajectory, reset_bounding_box=False)
        if len(updated_poses) > 1:
            updated_trajectory = get_trajectory(updated_poses, [0, 1, 0])
            # vis.add_geometry(updated_trajectory, reset_bounding_box=False)
            updated_trajectory.add_line(vis)

        # for kf in keyframes:
        #     camera = create_camera_actor(0.9)
        #     camera.transform(kf)
        #     vis.add_geometry(camera, reset_bounding_box=False)
        camera = create_camera_actor2(0.9, pose, 0.1)
        camera.add_line(vis)
        # camera.transform(pose)
        # vis.add_geometry(camera, reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()
        frame_count += 1


    def update_mesh_and_voxels_continuous(vis):
        global max_frame, frame_rate, play_start
        if play_start:
            return
        play_start = True
        print("start playing contiuously")
        for i in range(max_frame):
            t1 = time()
            update_mesh_and_voxels(vis)
            time_lapse = time() - t1
            if time_lapse < 1 / frame_rate:
                sleep((1 / frame_rate) - time_lapse)
        print("end playing contiuously")


    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback(65, update_mesh_and_voxels)
    vis.register_key_callback(66, update_mesh_and_voxels_continuous)
    vis.run()
    vis.destroy_window()
