import numpy as np
import polyscope as ps
from utils import read_off, to_np, bounding_box_diag
from task1_constraints import sample_constraints
from tests import check_constraints
from task2_solver import generate_grid
from task2_solver import local_predictor
from utils import vals2colors
from skimage.measure import marching_cubes
from tests import test_make_screens

def main(data_path):
    # read data
    data_path = './data/cat.off'
    vertices, faces_gt, normals = read_off(data_path)
    
    ### Data Visualization
    ps.init() # Initialization. Don't need to reinitialize each time
    if 'cat' in data_path:
        ps.set_up_dir("z_up") # Fix default camera orientation 
    gt_mesh = ps.register_surface_mesh("gt_mesh", vertices, faces_gt)
    gt_mesh.set_enabled(True)

    ps_cloud = ps.register_point_cloud('pts', vertices)
    ps_cloud.add_vector_quantity("Normal vec", normals, radius=0.01, length=0.02, color=(0.2, 0.5, 0.5), enabled=True)

    ps.show() # a window with visualization should be opened.
    
    ### Sample Constraints
    bbox_diag = bounding_box_diag(vertices)
    eps = bbox_diag * 0.01 # define eps parameter for sampling = 0.01 * bounding box diagonal
    new_verts, new_vals = sample_constraints(vertices, normals, eps)

    all_pts = np.concatenate([vertices, new_verts])
    all_vals = np.concatenate([np.zeros(len(vertices)), new_vals])

    # Sampled constraints visualization
    ps.register_point_cloud('pos pts', all_pts[all_vals>0])
    ps.register_point_cloud('neg pts', all_pts[all_vals<0])

    ps.show()
    
    ### Generate Grid
    resolution = 30
    grid_pts, coords_matrix = generate_grid(all_pts, resolution)
    local_radius = bbox_diag * 0.1

    ps.register_point_cloud('grid pts', grid_pts, radius=0.001)
    ps.show()
    
    ### Evaluate implicit function at the grid points
    print("test")
    pred_vals = local_predictor(
            grid_pts=grid_pts,
            constr_pts=all_pts,
            constr_vals=all_vals,
            local_radius=local_radius,
            degree=1,
            reg_coef=1)

    colors = vals2colors(pred_vals) # map implicit value to color for visualization
    grid_cloud = ps.register_point_cloud('grid pts', grid_pts, radius=0.001)
    grid_cloud.add_color_quantity("rand colors", colors, enabled=True)
    ps.show()
    
    ### Surface extraction
    verts, faces, _, _ = marching_cubes(pred_vals.reshape([resolution, resolution, resolution]), level=0)
    verts = (coords_matrix[:3, :3] @ verts.T + coords_matrix[:3, 3:4]).T

    pred_mesh = ps.register_surface_mesh("mesh", verts, faces)
    ps.show()

if __name__ == "__main__":
    main(data_path='./data/cat.off')
    # test_make_screens() <- when ready run only this (without main(...))