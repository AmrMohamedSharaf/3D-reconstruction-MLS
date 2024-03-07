# Surface reconstruction using Moving Least Squares method

## Problem 

Given an oriented point cloud data set S = {pi , ni}, we want to retrieve the surfce:
* 1. * -The implicit representation of the 3D model has constrains
* 1. * -Point cloud data might contain noise and outliers
* 1. * -For any S in F(x) where F(x) is the implicit function S must be guaranteed to be a manifold

**Step-1 Constrains Satisfaction :** We construction an implicit function F(x) with following conditions 1- For point x on the surface, the value of F(x) = 0 2- for any x = s+α for a point s on the surface , F(x) = s+α

**Step-2 Implicit Surface ** : Given our implicit surface constraints , we can now build a system of linear equations for any point on the surface. this model can be represented as the sum over c =1 , wc⋅ϕ(∥x−c∥^2)

ϕ is a kernel function. The weights w are used in a weighted least squares fitting process, where the contributions of nearby points are emphasized while those of more distant points are diminished. in our case ϕ is the windland function with center c. Thus the further points are for the center c the less dominant their entry. Thus our system is diagonally dominant

**Step 3 - Iso Surfacing : ** Using the marching cube algorithm , we can retrieve the mesh 

# Results

### Input
<img src="https://github.com/AmrMohamedSharaf/3D-reconstruction-MLS/assets/69557495/5a9aa2c7-1b19-4b46-998e-81dfbcedfcff" width="300" height="300">

### Sampled Points
<img src="https://github.com/AmrMohamedSharaf/3D-reconstruction-MLS/assets/69557495/c95cf41f-c02b-4e3c-9912-079dbdc9a677" width="300" height="300">

### Reconstructed Mesh 
<img src="https://github.com/AmrMohamedSharaf/3D-reconstruction-MLS/assets/69557495/3c65e047-7da7-429d-8a36-73d61cf35ccf" width="300" height="300">









