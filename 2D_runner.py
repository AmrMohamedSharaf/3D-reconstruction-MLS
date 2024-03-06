#%%
import numpy as np
import polyscope as ps
from utils import read_off, to_np, bounding_box_diag
import matplotlib.pyplot as plt
from data.lines_generator_2D import sample_cirlce 
from utils import plot_arrow
from task2_solver import generate_grid
from task1_constraints import sample_constraints
from tests import check_constraints
from task2_solver import global_predictor
from skimage import measure
from data.lines_generator_2D import sample_curved_cirlce
from task2_solver import local_predictor


def plot_points_normals(vertices, normals):
    plt.figure(figsize=(5,5))
    plt.scatter(vertices[:,0], vertices[:,1])
    for vert, norm in zip(vertices, normals):
        plot_arrow(begin=vert, end=vert+norm*0.3)
    plt.gca().set_xlim(-1.4, +1.4)
    plt.gca().set_ylim(-1.4, +1.4)
    plt.title('Input data ')
    plt.show()

def show_constraint_points(pts, vals):
    plt.scatter(pts[vals==0][:,0], pts[vals==0][:,1], label='surface pts')
    plt.scatter(pts[vals>0][:,0], pts[vals>0][:,1], label='outside pts')
    plt.scatter(pts[vals<0][:,0], pts[vals<0][:,1], label='inside pts')
    plt.legend()
    plt.axis('square')

def show_grid_pts(grid_pts, grid_vals):
    plt.scatter(grid_pts[grid_vals>=0][:,0], grid_pts[grid_vals>=0][:,1], s=5, label='grid outside')
    plt.scatter(grid_pts[grid_vals<0][:,0], grid_pts[grid_vals<0][:,1], s=5, label='grid inside')
    plt.legend()
    plt.axis('square')

def transform_pts(pts, matr):
    return (matr[:2, :2] @ pts.T + matr[:2, 2:3]).T

def reconstruct(pred_vals, resolution):
    return measure.find_contours(pred_vals.reshape(resolution, resolution), 0)

def show_reconstruction(vertices, pred_vals, resolution, transform_matrix):
    contours = reconstruct(pred_vals, resolution)

    ax = plt.gca()
    ax.scatter(vertices[:,0], vertices[:,1], label='init vertices')
 
    for contour in contours:
        contour = transform_pts(contour, transform_matrix)
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='b')
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='b', label='reconstruction')

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()
    plt.axis('square')

def main(vertices, normals, model='global'):
    # %% PART 1: Setting up constraints
    bbox_diag = bounding_box_diag(vertices)
    eps_mul = np.linspace(0.01, 1, 10)[1]
    eps = bbox_diag * 0.05
    # TODO: implement sample_constraints in task1_constraints.py
    new_verts, new_vals = sample_constraints(vertices, normals, eps)

    constr_pts = np.concatenate([vertices, new_verts])
    constr_vals = np.concatenate([np.zeros(len(vertices)), new_vals])

    ## quick check that implementation is correct
    #check_constraints(vertices, normals, constr_pts, constr_vals)

    plt.figure(figsize=(5,5))
    show_constraint_points(constr_pts, constr_vals)
    for vert, norm in zip(vertices, normals):
        plot_arrow(begin=vert, end=vert+norm*0.15)
    plt.title('Input data + constraints with given 2*eps')
    plt.show()

    # %% PART 2.1: Grid Generation
    resolution = 20
    grid_pts, coords_matrix = generate_grid(constr_pts, resolution, num_dims=2)

    plt.figure(figsize=(5,5))
    show_constraint_points(constr_pts, constr_vals)
    plt.scatter(grid_pts[:,0], grid_pts[:,1], s=2, label='grid')
    plt.legend()
    plt.axis('square')
    plt.title('Sampled grid points')
    plt.show()

    # %% PART 2.2: Setting up predictor
    if model == 'global':
        # TODO: implement global_predictor in task2_solver.py
        pred_vals = global_predictor(grid_pts, constr_pts, constr_vals, degree=2)
    elif model == 'local':
        # TODO: implement local_predictor in task2_solver.py
        local_radius = bbox_diag * 0.2
        pred_vals = local_predictor(grid_pts, constr_pts, constr_vals, local_radius, degree=2, reg_coef=0)
    elif model == 'local+reg':
        # TODO: implement local_predictor + regularization in task2_solver.py
        local_radius = bbox_diag * 0.2
        pred_vals = local_predictor(grid_pts, constr_pts, constr_vals, local_radius, degree=2, reg_coef=0.001)

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    show_constraint_points(constr_pts, constr_vals)
    plt.title('Sample constraints')
    plt.subplot(122)
    show_grid_pts(grid_pts, pred_vals)
    plt.title('Evaluated grid nodes')
    plt.show()
    
    # %% PART 3: Reconstruction
    plt.figure(figsize=(5,5))
    show_reconstruction(vertices, pred_vals, resolution, coords_matrix)
    plt.title('Reconstruction')
    plt.show() 
    # %%

if __name__ == "__main__":
    # lets try global model first with simple data
    verts, norms = sample_cirlce(1, 20)
    plot_points_normals(verts, norms)
    main(verts, norms, model='global')
   
   # more complex data
    verts, norms = sample_curved_cirlce(1, 50)
    plot_points_normals(verts, norms)
    main(verts, norms, model='global')

    # now lets try local model
    main(verts, norms, model='local')

    # adding regularization should help
    main(verts, norms, model='local+reg')
