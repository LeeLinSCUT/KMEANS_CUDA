import h5py 
import numpy as np 
num = 512
scale_size = 0.5
x1 = np.random.normal(loc=0,scale=scale_size,size=num)
y1 = np.random.normal(loc=1,scale=scale_size,size=num)
z1 = np.random.normal(loc=1,scale=scale_size,size=num)
x1 = np.expand_dims(x1,axis=1)
y1 = np.expand_dims(y1,axis=1)
z1 = np.expand_dims(z1,axis=1)

x2 = np.random.normal(loc=0,scale=scale_size,size=num)
y2 = np.random.normal(loc=1,scale=scale_size,size=num)
z2 = np.random.normal(loc=-1,scale=scale_size,size=num)
x2 = np.expand_dims(x2,axis=1)
y2 = np.expand_dims(y2,axis=1)
z2 = np.expand_dims(z2,axis=1)

x3 = np.random.normal(loc=0,scale=scale_size,size=num)
y3 = np.random.normal(loc=-1,scale=scale_size,size=num)
z3 = np.random.normal(loc=1,scale=scale_size,size=num)
x3 = np.expand_dims(x3,axis=1)
y3 = np.expand_dims(y3,axis=1)
z3 = np.expand_dims(z3,axis=1)

x4 = np.random.normal(loc=0,scale=scale_size,size=num)
y4 = np.random.normal(loc=-1,scale=scale_size,size=num)
z4 = np.random.normal(loc=-1,scale=scale_size,size=num)
x4 = np.expand_dims(x4,axis=1)
y4 = np.expand_dims(y4,axis=1)
z4 = np.expand_dims(z4,axis=1)
f = h5py.File('data.h5py','w')

xyz1 = np.concatenate((x1,y1,z1),axis=-1)
xyz2 = np.concatenate((x2,y2,z2),axis=-1)
xyz3 = np.concatenate((x3,y3,z3),axis=-1)
xyz4 = np.concatenate((x4,y4,z4),axis=-1)

xyz = np.concatenate((xyz1,xyz2,xyz3,xyz4),axis=0)

d1 = f.create_dataset('data',data=xyz)

a = np.array([[0,0.2,0.2],[0,0.2,-0.8],[0,-0.6,0.2],[0,-0.2,-0.2]])

d2 = f.create_dataset('init_xyz',data=a)


