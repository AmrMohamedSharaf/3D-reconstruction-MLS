import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from skimage import measure
from utils import generate_grid , bounding_box_diag , transform_to_polynomial_basis



def sample_constraints(vertices, normals, eps):
    # getting the shape of the vertex arrays 
    n_points = vertices.shape[0]
    new_vert_pos, new_values_pos = np.zeros((n_points , vertices.shape[1])), np.zeros(n_points)
    new_vert, new_values = np.zeros((n_points*2 , vertices.shape[1])), np.zeros(n_points * 2)
    kdtree = KDTree(vertices) # nearest point search 
    new_vertices_pos = vertices + eps * normals #offset vectors for the first itr
    new_values_pos [:n_points] = eps
    # first loop for the positive offset 
    while True:
        # closest points querey 
        _,closest_point_indices_pos = kdtree.query(new_vertices_pos)
        violation_mask = closest_point_indices_pos != np.arange(n_points) # the corresponding ones should show a one to one relationship with verticees 
        if not np.any(violation_mask):
            break  # the mask doesn't catch any violationtns then exsit 
        new_vertices_pos[violation_mask] = vertices[violation_mask]+normals[violation_mask]*eps
        # if not then try for a lower eps value for all sampling points that were caught using the mask 
        new_vert_pos[np.where(violation_mask)]
        new_values_pos[violation_mask] = eps
        
        eps /= 2.0

    # same procedure but for negative offsets 
    new_vert_neg, new_values_neg = np.zeros((n_points , vertices.shape[1])), np.zeros(n_points)
    new_vert, new_values = np.zeros((n_points*2 , vertices.shape[1])), np.zeros(n_points * 2)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    kdtree = KDTree(vertices)
    new_vert_neg = vertices - eps * normals
    new_values_neg [:n_points] = -eps
    
    while True:
        _,closest_point_indices_neg = kdtree.query(new_vert_neg)
        violation_mask = closest_point_indices_neg != np.arange(n_points)
        if not np.any(violation_mask):
            break  
        new_vert_neg[violation_mask] = vertices[violation_mask]-normals[violation_mask]*eps

        new_values_neg[violation_mask] = -eps
        
        eps /= 2.0

    new_vert[:n_points] = new_vertices_pos
    new_vert[n_points:] = new_vert_neg
    new_values[:n_points] = new_values_pos
    new_values[n_points:] = new_values_neg
    return new_vert, new_values



def global_predictor(grid_pts, constr_pts, constr_vals, degree=2):
    """Evaluate implicit function in space

    Args:
        grid_pts (np.array [N, 3]): 3D coordinates of N points in space. Grid points
        constr_pts (np.array [M, 3]): 3D coordinates of M points in space. Constraints points
        constr_vals (np.array, [M, 1]): constraint values defined on constr_pts
        degree (int): degree of Polynomial

    Returns:
        pred_vals (np.array [N, 1]): implicit function values for each of the grid points
    """
    pred_vals = np.zeros((len(grid_pts), 1))

    nearby_vertices_poly = transform_to_polynomial_basis(constr_pts, degree)
    grid_pt_poly = transform_to_polynomial_basis(grid_pts, degree)
    coefficients = np.linalg.solve(nearby_vertices_poly.T @ nearby_vertices_poly, nearby_vertices_poly.T @ constr_vals)
    pred_vals = grid_pt_poly @ coefficients
  
    
    return pred_vals.flatten()




from utils import eval_grid_point
def local_predictor(grid_pts, constr_pts, constr_vals, local_radius, degree, reg_coef=0):
    """Evaluate implicit function in space

    Args:
        grid_pts (np.array [N, 2 or 3]): 2D/3D coordinates of N points in space. Grid points
        constr_pts (np.array [M, 2 or 3]): 2D/3D coordinates of M points in space. Constraints points
        constr_vals (np.array, [M, 1]): constraint values defined on constr_pts
        local_radius (float): parameter to set the weight function / neighbourhood radius
        degree (int): degree of Polynomial
        reg_coef (optional) (float): regularization parameter

    Returns:
        pred_vals (np.array [N, 1]): implicit function values for each of the grid points
    """
    tree = KDTree(constr_pts)
    pred_vals = np.zeros(len(grid_pts))
    for cur_grid_idx, g_pt in enumerate(grid_pts):
        nearby_indices = tree.query_ball_point(g_pt,local_radius)
        need_neighbours = degree
        closest_pts, closest_vals = constr_pts[nearby_indices], constr_vals[nearby_indices]

        if len(closest_pts) < need_neighbours:
            pred_vals[cur_grid_idx] = 1000
        else:
            pred_vals[cur_grid_idx] = eval_grid_point(g_pt, closest_pts, closest_vals, local_radius, degree, reg_coef)
    return pred_vals




def full_reconstruction(data, resolution, predictor_type, degree=2, reg_coef=0., num_dims=2, eps_mul=0.1, radius_mul=0.1):
    vertices, normals = data['verts'], data['normals']
    bbox_diag = bounding_box_diag(vertices)
    eps = bbox_diag * eps_mul
    new_verts, new_vals = sample_constraints(vertices, normals, eps)

    constr_pts = np.concatenate([vertices, new_verts])
    constr_vals = np.concatenate([np.zeros(len(vertices)), new_vals])

    grid_pts, coords_matrix = generate_grid(constr_pts, resolution, num_dims=num_dims)

    if predictor_type == 'global':
        pred_vals = global_predictor(grid_pts, constr_pts, constr_vals, degree=degree)
        
        
        # plt.figure(figsize=(10,5))
        # plt.subplot(121)
        # show_constraint_points(constr_pts, constr_vals)
        # plt.title('Sample constraints')
        # plt.subplot(122)
        # show_grid_pts(grid_pts, pred_vals)
        # plt.title('Evaluated grid nodes')
        # plt.show()

    elif predictor_type == 'local':
        local_radius = bbox_diag * radius_mul
        pred_vals = local_predictor(grid_pts, constr_pts, constr_vals, local_radius, degree=degree, reg_coef=reg_coef)
    else:
        assert 0, 'unknown model type'

    if num_dims == 2:
        contours = measure.find_contours(pred_vals.reshape(resolution, resolution), 0)
        if len(contours) == 1:
            verts = contours[0]
        elif len(contours) == 0:
            return np.array([]), np.array([])
        else:
            verts = np.concatenate(contours, axis=0)
        verts = (coords_matrix[:2, :2] @ verts.T + coords_matrix[:2, 2:3]).T
        faces = np.array([])

    elif num_dims == 3:
        if np.unique(pred_vals).shape[0] == 1:
            verts, faces = np.array([]), np.array([])
        else:
            verts, faces, _, _ = measure.marching_cubes(pred_vals.reshape([resolution, resolution, resolution]), level=0)
            verts = (coords_matrix[:3, :3] @ verts.T + coords_matrix[:3, 3:4]).T
    else:
        assert 0, 'unknown num dims'
    
    return verts, faces