import numpy as np 
import h5py 
f = h5py.File('data.h5py','r')
data = f["data"][:][:]
np.savetxt("data.xyz",data,'%.8f')

