import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt



def transform_to_polynomial_basis(pts, degree):
    """ Represent 2D/3D points (pts) in polynomial basis of given degree
    e.g. degree 2: (x, y) -> (1, x, y, x^2, y^2, xy)
    
    Args:
        pts (np.array [N, 2 or 3]): 2D/3D coordinates of N points in space
        degree (int): degree of Polynomial

    Returns:
        ext_pts (np.array [N, ?]): points (pts) in Polynomial basis of given degree. 
        The second dimension depends on the polynomial degree and initial dimention of points (2D or 3D)  
    """
    if degree == 0:
        return np.ones([len(pts), 1])
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    ext_pts = np.concatenate([np.ones([len(pts), 1]), x, y], axis=1)
    for i in range(2, degree + 1):
        for j in range(i + 1):
            term = (x ** (i - j)) * (y ** j)
            ext_pts = np.concatenate([ext_pts, term], axis=1)
    return ext_pts


    plt.scatter(pts[vals==0][:,0], pts[vals==0][:,1], label='surface pts')
    plt.scatter(pts[vals>0][:,0], pts[vals>0][:,1], label='outside pts')
    plt.scatter(pts[vals<0][:,0], pts[vals<0][:,1], label='inside pts')
    plt.legend()
    plt.axis('square')

    """Generate grid over the point cloud with given resolution

    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution

    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    if num_dims == 3:
        coords = np.mgrid[:res, :res, :res]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        length += length/res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[2, 2] = length[2] / res
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
        coords = coords.T
    elif num_dims==2:
        coords = np.mgrid[:res, :res]
        coords = coords.reshape(2, -1)
        coords_matrix = np.eye(3)
        length = b_max - b_min
        length += length/res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[0:2, 2] = b_min
        coords = np.matmul(coords_matrix[:2, :2], coords) + coords_matrix[:2, 2:3]
        coords = coords.T
    else:
        assert 0
    return coords, coords_matrix


def eval_grid_point(eval_pt, local_constr_pts, local_constr_vals, local_radius, degree, reg_coef):



    """Evaluate implicit function at the eval point

    Args:
        eval_pt (np.array [2 or 3,]): 2D/3D coordinate of a point in space. A grid point
        local_constr_pts (np.array [K, 2 or 3]): 2D/3D coordinates of M points in space (constrain points)
        local_constr_vals (np.array, [K, 1]): constraint values defined on local_constr_pts
        local_radius (float): parameter to set the weight function
        degree (int): degree of Polynomial
        reg_coef (optional) (float): regularization parameter

    Returns:
        pred_val (float): implicit function value at the point
    """
    pred_val = 0
    nearby_vertices_poly = transform_to_polynomial_basis(local_constr_pts, degree)
    point = []
    point.append(eval_pt)
    grid_pt_poly = transform_to_polynomial_basis(np.array(point), degree)
    
    weights = wendland(np.linalg.norm(eval_pt - local_constr_pts, axis=1), local_radius)
 
    # debugging lines 

    # coefs = np.linalg.solve(nearby_vertices_poly.T @ nearby_vertices_poly, nearby_vertices_poly.T @ local_constr_vals)
    # print(nearby_vertices_poly.shape)
    # pred_val = grid_pt_poly @ coefs
    
    # regulrization step
    # weights are squared for normlization 
    #_____________________________________________________________
    W = np.diag(weights)
    _, p = nearby_vertices_poly.shape
    sqrt_weights = np.sqrt(weights)
    awaighted = nearby_vertices_poly * sqrt_weights[:, np.newaxis]
    y_weighted = local_constr_vals * sqrt_weights
    #_________________________________________________________________
    identity_matrix = np.identity(p) # following the formaula from lecture 
    ridge_term = reg_coef * identity_matrix
    coefficients = np.linalg.inv(awaighted.T @ awaighted + ridge_term) @ awaighted.T @ y_weighted


    pred_val =  np.dot(grid_pt_poly, coefficients)
    return pred_val



def wendland(r, h):
    """Wendland weight function: (1 - r/h)^4 * (4 * r/h + 1); if r>=h -> weight=0

    Args:
        r (np.array [N] or float): distance parameter
        h (float): weight parameter

    Returns:
        weights (np.array [N, 1] or float): weight function values
    """
    assert h >= 0
    if isinstance(r, float) or isinstance(r, int):
        assert r>=0
    else:
        assert (r>=0).all()

    x = r/h
    weights = (1-x)**4 * (4 * x + 1)
    if isinstance(r, float) or isinstance(r, int):
        if r >= h:
            weights = 0
    else:
        weights[r >= h] = 0
    return weights



    """Generate grid over the point cloud with given resolution

    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution

    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    if num_dims == 3:
        coords = np.mgrid[:res, :res, :res]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        length += length/res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[2, 2] = length[2] / res
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
        coords = coords.T
    elif num_dims==2:
        coords = np.mgrid[:res, :res]
        coords = coords.reshape(2, -1)
        coords_matrix = np.eye(3)
        length = b_max - b_min
        length += length/res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[0:2, 2] = b_min
        coords = np.matmul(coords_matrix[:2, :2], coords) + coords_matrix[:2, 2:3]
        coords = coords.T
    else:
        assert 0
    return coords, coords_matrix


def bounding_box_diag(pts):
    b_min_g = np.min(pts, axis=0)
    b_max_g = np.max(pts, axis=0)
    diag = np.linalg.norm(b_max_g - b_min_g)
    return diag



def vals2colors(vals):
    colors = np.ones([len(vals), 3])
    colors[vals < 0] = np.array([1,0,0])
    colors[vals > 0] = np.array([0,1,0])
    colors[vals>=100] = np.array([0,0,0])
    return colors



def generate_grid(point_cloud, res, num_dims=3):
    """Generate grid over the point cloud with given resolution

    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution

    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    if num_dims == 3:
        coords = np.mgrid[:res, :res, :res]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        length += length/res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[2, 2] = length[2] / res
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
        coords = coords.T
    elif num_dims==2:
        coords = np.mgrid[:res, :res]
        coords = coords.reshape(2, -1)
        coords_matrix = np.eye(3)
        length = b_max - b_min
        length += length/res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[0:2, 2] = b_min
        coords = np.matmul(coords_matrix[:2, :2], coords) + coords_matrix[:2, 2:3]
        coords = coords.T
    else:
        assert 0
    return coords, coords_matrix