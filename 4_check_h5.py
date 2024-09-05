# open the file as 'f'

import numpy as np
import h5py

f = h5py.File("/home/fovea/Desktop/Saeid_3080/new_data/modelnet40_ply_hdf5_2048/ply_data_test1.h5")

data = f['data'][:].astype('float32')
label = f['label'][:].astype('int64')
print(data.shape, type(data))
print(label.shape, type(label))
f.close()


'''
import numpy as np
import h5py
 
data = np.random.rand(2048,2048,3)
label = np.random.randn(2048,1)
 
with h5py.File('ply_data.h5', 'w') as f:
    f.create_dataset('data', data = data)
    f.create_dataset('label', data = label)
'''