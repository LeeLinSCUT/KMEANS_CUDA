import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
test_module=tf.load_op_library(os.path.join(BASE_DIR, 'kmeans.so'))

def kmeans(xyz,init_xyz):
    return test_module.kmeans(xyz,init_xyz)
ops.NoGradient('QueryBallPoint')

if __name__=='__main__':
	import numpy as np
	import h5py

	f = h5py.File("data.h5py",'r')
	data = f["data"][:][:]  #(1024,3)
	init_xyz = f["init_xyz"][:][:]

	with tf.device('/gpu:0'):
		xyz = tf.constant(data) 
		init_xyz = tf.constant(init_xyz) 
		xyz = tf.expand_dims(xyz,0)
		init_xyz = tf.expand_dims(init_xyz,0)
		xyz = tf.to_float(xyz)
		init_xyz = tf.to_float(init_xyz)
		result = kmeans(xyz,init_xyz)

	with tf.Session() as sess:
		result = result.eval()
		result = result.astype(float)
		idx=0
		for i in result[0][:]:
			if(i==1):
				result[0][idx] = 0.3
			if(i==2):
				result[0][idx] = 0.6
			if(i==3):
				result[0][idx] = 0.9
			idx +=1
		result = result.reshape(2048,1)
		print(result)
		np.savetxt("result",result,fmt='%.2f')

		data =data.reshape(2048,3)
		data = np.concatenate((data,result),axis=-1)
		data = np.concatenate((data,result),axis=-1)
		data = np.concatenate((data,result),axis=-1)
		np.savetxt("data.xyzrgb",data,fmt='%.8f') 

		init_xyz = init_xyz.eval()
		init_xyz =init_xyz.reshape(4,3)
		np.savetxt("init_xyz",init_xyz,fmt='%.8f')