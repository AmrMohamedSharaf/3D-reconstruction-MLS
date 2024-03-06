import os
import numpy as np
from scipy.spatial import KDTree
from utils import read_off, bounding_box_diag
from task1_constraints import sample_constraints
from data import lines_generator_2D
from task2_solver import full_reconstruction
from utils import mesh_distance, calc_grid_cell_diag

model_names = [ 'bunny-1000.off',  'sphere.off', 'cat.off', ]

def test_constraints_sampling_3D(data_dir='./data'):
    for cur_name in model_names:
        vertices, _, normals = read_off(f"{data_dir}/{cur_name}")
        diag = bounding_box_diag(vertices)
        for eps in diag * np.array([10, 1, 0.1, 0.01]):
            new_verts, new_vals = sample_constraints(vertices, normals, eps)
            check_constraints(vertices, normals, new_verts, new_vals)

def test_constraints_sampling_2D():
    for r in range(1, 10, 2):
        for N in range(20, 50, 10):
                for eps_mul in np.linspace(0.01, 1, 10):
                    vertices, normals = lines_generator_2D.sample_cirlce(r, N)

                    bbox_diag = bounding_box_diag(vertices)
                    eps = bbox_diag * eps_mul
                    new_verts, new_vals = sample_constraints(vertices, normals, eps)

                    constr_pts = np.concatenate([vertices, new_verts])
                    constr_vals = np.concatenate([np.zeros(len(vertices)), new_vals])

                    check_constraints(vertices, normals, constr_pts, constr_vals)

def check_constraints(vertices, normals, pts, vals):
    N = vertices.shape[0]
    pos_verts = pts[vals > 0]
    neg_verts = pts[vals < 0]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    assert N == pos_verts.shape[0] == neg_verts.shape[0]

    tree = KDTree(vertices)

    dists_p, idx_p = tree.query(pos_verts, k=1)
    dists_n, idx_n = tree.query(neg_verts, k=1)

    pos_sort_p = np.argsort(idx_p)
    pos_sort_n = np.argsort(idx_n)
    pos_verts, dists_p, idx_p = pos_verts[pos_sort_p], dists_p[pos_sort_p], idx_p[pos_sort_p]
    neg_verts, dists_n, idx_n = neg_verts[pos_sort_n], dists_n[pos_sort_n], idx_n[pos_sort_n]

    assert np.allclose(idx_p, np.arange(N))
    assert np.allclose(idx_n, np.arange(N))

    assert np.allclose(dists_p, vals[vals>0])
    assert np.allclose(dists_n, -1*vals[vals<0])

    assert np.allclose(((pos_verts - vertices) * normals).sum(1), dists_p)
    assert np.allclose(-((neg_verts - vertices) * normals).sum(1), dists_n)

def test_global():
    resolution = 30
    for r in np.linspace(1, 100, 10):
        for N in range(10, 100, 20):
            print(r , N)
            vertices, normals = lines_generator_2D.sample_cirlce(r, N)
            pred, _ = full_reconstruction(
                {'verts': vertices, 'normals': normals},
                resolution=resolution,
                predictor_type='global',
                degree=2, num_dims=2, eps_mul=0.1)

            thr = calc_grid_cell_diag(vertices, resolution)
            if pred.shape[0] == 0:
                assert 0, "Prediction is empty"
            dist = mesh_distance(vertices, _, pred, _)
            assert dist < thr

def test_local():
    resolution = 50
    for r in np.linspace(0.7, 10, 5):
        for N in range(20, 100, 30):
            vertices, normals = lines_generator_2D.sample_cirlce(r, N)
            radius_mul = 0.2 if N <= 20 else 0.1
            pred, _ = full_reconstruction(
                {'verts': vertices, 'normals': normals},
                resolution=resolution,
                predictor_type='local',
                degree=2, reg_coef=0.001, num_dims=2, eps_mul=0.1, radius_mul=radius_mul)

            thr = calc_grid_cell_diag(vertices, resolution)
            if pred.shape[0] == 0:
                assert 0, "Prediction is empty"
            dist = mesh_distance(vertices, _, pred, _)
            assert dist < thr
    print("testing")

def test_3d():
    # import polyscope as ps

    resolution = 30
    for cur_name in model_names:
        vertices, faces_gt, normals = read_off(f"./data/{cur_name}")

        res_verts, res_faces = full_reconstruction(
            {'verts': vertices, 'normals': normals},
            resolution,
            predictor_type='local',
            degree=1, reg_coef=1, num_dims=3, eps_mul=0.02, radius_mul=0.1)

        # ps.init()
        # ps.register_surface_mesh("mesh", vertices, faces_gt)
        # # ps.register_surface_mesh("mesh pred", res_verts, res_faces)
        # ps.show()

        thr = calc_grid_cell_diag(vertices, resolution)
        if res_verts.shape[0] == 0:
                assert 0, "Prediction is empty"
        dist = mesh_distance(vertices, faces_gt, res_verts, res_faces)
       
        # assert dist < thr

def test_make_screens():
    import polyscope as ps
    os.makedirs('./bonus_img_saves', exist_ok=True)
    resolution = 30
    for cur_name in model_names:
        vertices, faces_gt, normals = read_off(f"./data/{cur_name}")

        bbox_diag = bounding_box_diag(vertices)
        res_verts, res_faces = full_reconstruction(
            {'verts': vertices, 'normals': normals},
            resolution,
            predictor_type='local',
            degree=1, reg_coef=1, num_dims=3, eps_mul=0.02, radius_mul=0.1)

        ps.init()
        pred_mesh = ps.register_surface_mesh("mesh pred", res_verts, res_faces)
        pred_mesh.set_enabled(True)
        center = vertices.mean(axis=0)

        view_pts = np.random.rand(5,3)
        view_pts = view_pts / np.linalg.norm(view_pts, axis=1)[:,None] * bbox_diag * 1.2
        view_pts += center
        for idx, view in enumerate(view_pts):
            ps.look_at(view, center)
            ps.screenshot(f"./bonus_img_saves/{cur_name.split('.')[0]}_img{idx}.jpg")

        ps.look_at([0, bbox_diag, bbox_diag], center)
        ps.screenshot(f"./bonus_img_saves/{cur_name.split('.')[0]}_img{idx+1}.jpg")

    


# def main():
#     test_3d()
# main()