import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def to_np(a_list):
    return np.array(a_list)

def read_off(file_name: str):
    # Open the file for reading
    with open(file_name, 'r') as file:
        # Read the first line and check if it is 'OFF'
        if file.readline().strip() != 'NOFF':
            raise ValueError('Invalid file format')
        
        # Read the next two lines to get the number of vertices and faces
        num_vertices, num_faces, _ = map(int, file.readline().strip().split())
        
        # Read the vertices
        vertices_normals = [list(map(float, file.readline().strip().split())) for _ in range(num_vertices)]
        vertices, normals = zip(*[[x[:3], x[3:]] for x in vertices_normals])
        
        # Read the faces
        faces = [list(map(int, file.readline().strip().split()))[1:] for _ in range(num_faces)]
        
        # Return the vertices, faces and normals
        return to_np(vertices), to_np(faces), to_np(normals)

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

def numpy_equal(a, b, thresh=0.00001):
  return (np.abs(a-b)<thresh).all()

def mesh_distance(verts1, faces1, verts2, faces2):
    return pseudo_hausdorff(verts1, verts2)

def pseudo_hausdorff(verts1, verts2):
    # verts1 -> verts2 distance
    recon_tree = KDTree(verts2)
    dists, _ = recon_tree.query(verts1, k=1)
    return np.max(dists)

def calc_grid_cell_diag(vertices, res):
    b_min = np.min(vertices, axis=0)
    b_max = np.max(vertices, axis=0)
    thr = np.sqrt(np.sum(((b_max - b_min)/res)**2))
    return thr

def plot_arrow(end, begin=(0, 0), color="black", lw=2, zorder=0):
  prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", shrinkA=0, shrinkB=0, color=color, lw=lw)
  return plt.annotate("", xy=end, xytext=begin, arrowprops=prop, zorder=zorder)

def make_plot_squared():
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')