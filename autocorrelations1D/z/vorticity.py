import numpy as np
def omg(u_r, u_th, u_z, r, th, z):
  omgR = np.zeros((u_r.shape))
  omgTh= np.zeros((u_r.shape))
  omgZ = np.zeros((u_r.shape))
  # constructing an array of dimension(r) filled with zeros 
  r3d  = np.tile(r, (len(z), len(th), 1)).T # Changing a 1D array to a 3D array as we have to divide u(:,:,:) by r. 
                                            # In python we have to reshape our array to 3D to perform the division.
  # increment for spatial derivatives
  dth = th[1] - th[0]
  dz  =  z[1] -  z[0]

  # calculating velocity gradients
  dUthdr = np.gradient(u_th,r, axis=0)
  dUzdr  = np.gradient(u_r,dth,axis=0)

  dUrdth = (1.0/r3d)*np.gradient(u_r,dth,axis=1) - u_th/r3d
  dUzdth = (1.0/r3d)*np.gradient(u_z,dth,axis=1)

  dUrdz  = np.gradient(u_r ,dz,axis=2)
  dUthdz = np.gradient(u_th,dz,axis=2)

  #calculating components of vorticity vector
  omgR  = dUzdth - dUthdz
  omgTh = dUrdz  - dUzdr
  omgZ  = dUthdr - dUrdth
  return omgR, omgTh, omgZ
