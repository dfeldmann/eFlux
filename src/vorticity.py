 
 

def omegaCyl(v1, v2, v3, x1, x2, x3):
    # compute vorticity vector field, which is the curl of the velcoity vector
    # field. Here in a cylindrical co-ordinate system.
    # input: velocity vector field v and grid x with the following components:
    # v1 = u = u_r          x1 = x = r
    # v2 = v = u_theta      x2 = y = theta
    # v3 = w = u_z          x3 = z = z
    # output: vorticity vector field with the following components
    # omega_x1 = omega_x = omega_r
    # omega_x2 = omega_y = omega_theta
    # omega_x3 = omega_z = omega_z
    
    import numpy as np

    # set-up radial co-ordinate as 3d array with same shape as velocity
    r = np.tile(x1, (len(x3), len(x2), 1)).T
  
    # Lame parameters for cylindrical co-ordinates
    h1 = 1.0
    h2 = r
    h3 = 1.0

    # compute radial derivatives
    dv2dx1 = np.gradient(h2*v2, x1, axis=0)
    dv3dx1 = np.gradient(h3*v3, x1, axis=0)

    # compute azimuthal derivatives
    dv1dx2 = np.gradient(h1*v1, x2, axis=1)
    dv3dx2 = np.gradient(h3*v3, x2, axis=1)

    # compute axial derivatives
    dv1dx3 = np.gradient(h1*v1, x3, axis=2)
    dv2dx3 = np.gradient(h2*v2, x3, axis=2)
    
    # compute vorticity (curl) components
    omega_x1 = 1.0/(h2*h3) * (dv3dx2 - dv2dx3)
    omega_x2 = 1.0/(h1*h3) * (dv1dx3 - dv3dx1)
    omega_x3 = 1.0/(h1*h2) * (dv2dx1 - dv1dx2)

    return omega_x1, omega_x2, omega_x3
