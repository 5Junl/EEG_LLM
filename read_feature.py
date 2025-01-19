import os
import glob
import scipy.io as sio

folder_path = "EEG_preprocessed"
mat_files = glob.glob(os.path.join(folder_path, '*.mat'))[:20]

for file_path in mat_files:
    data = sio.loadmat(file_path)
    #print(f"Loaded: {file_path}, keys: {list(data.keys())}")
    print(data['1'].shape)
    print(data['2'].shape)
    print(data['3'].shape)