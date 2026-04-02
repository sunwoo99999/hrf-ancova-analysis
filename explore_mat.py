import scipy.io as sio
import numpy as np

mat = sio.loadmat('dataset/SUB701_hrf.mat')
data = mat['hrf_ROI']
print('hrf_ROI shape:', data.shape)
print('hrf_ROI dtype:', data.dtype)

inner = data[0, 0]
print('inner type:', type(inner))
print('inner length:', len(inner))

for i, item in enumerate(inner):
    shape = getattr(item, 'shape', 'N/A')
    dtype = getattr(item, 'dtype', 'N/A')
    print(f'  item[{i}]: type={type(item).__name__}, shape={shape}, dtype={dtype}')
    if hasattr(item, 'shape') and item.size < 30:
        print(f'    value: {item}')
    elif hasattr(item, 'shape'):
        print(f'    first few: {item.flat[:6]}')
