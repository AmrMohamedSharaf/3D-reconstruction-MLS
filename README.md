This algorithm solves the problemm of surface reconstruction using Moving Least Squares method 

**Probelm** : Given an oriented point cloud data set S = {pi , ni}, we want to retrieve the surfce 
- The implecit representation of the 3D model has constratins
- Point cloud data might contain noise and outliers
- For any S in F(x) where F(x) is the implicit function  S must be guranteed to be a manifold


  Step1 Constratins Satisfaction :
  We construction an implict function F(x) with following conditions
  1- For point x on the surface, the value of F(x)  = 0
  2- for any x = s+α for a point s on the surface , F(x) = s+α

  Step2 :
  Given our implicit surface constrants , we can now build a system of linear equations for any point on the surfce.
  this model can be represented as the sum over c =1 , wc⋅ϕ(∥x−c∥^2)

  ϕ is a kernel function. The weights w are used in a weighted least squares fitting process, where the contributions of nearby points are emphasized while those of more distant points are diminished.
  in our case ϕ is the windland function with center c. Thus the further points are for the center c the less dominant their entry. Thus our system is diagonally dominant


