import torch
from torch.utils import data
import pandas as pd
import numpy as np
import json, os, glob
from sklearn.model_selection import train_test_split
from data_preparation.utils import _calculate_histogram
# from utils import _calculate_histogram
from data_preparation.utils_deeplearning import *



class YDataset_Hist(data.Dataset):
    def __init__(self, npy_path, label_path, norm_path,
                  lookup=None, mode=None, seed=0, start_doy_idx=11, 
                  end_doy_idx=38, feature_idx =list(range(15))):
        """
        
        Args:
            npy_path (str): path to the main folder containing .npy files
            target (str): column name of the yield column
            lookup (list): specifies list of years to filter
            seed(int): for splitting
            start_idx(int): lower bound of time series
            end_idx(int): upper bound of time series
            feature_idx(list): indices for subsetting at channel level

        """
        super(YDataset_Hist, self).__init__()

        self.npy_path = npy_path
        self.label_path = label_path
        self.norm_path = norm_path
        self.lookup = lookup
        self.mode = mode
        self.seed = seed
        self.start_doy_idx = start_doy_idx
        self.end_doy_idx = end_doy_idx
        self.feature_idx = feature_idx


         # load scaling
        if self.norm_path is not None:
            with open(self.norm_path, 'r') as file:
                self.norm = json.loads(file.read())

        # load yield
        with open(self.label_path) as file:
            self.labels = json.load(file)

        # ------------------prepare data for specific lookup
        data_indices = sorted(glob.glob(self.npy_path +'/**/*.npy',  recursive=True))  
        data_indices = [os.path.basename(f) for f in data_indices if any(str(xs) in f for xs in self.lookup)]

        # reconstruct filename as GEOID_YYYY to match yield format
        data_ids = [idx.split('.')[0][-5:] + '_'+ idx.split('.')[0][:4]for idx in data_indices]

        # get label keys
        labels_ids = [i for i in self.labels]
        
        # get intersection of pairs
        self.matched_pair =  list(set(labels_ids) & set(data_ids))
        self.geoid = np.array([x.split('_')[0] for x in self.matched_pair])

        # # -----------------------train_test split
        # stratify on state and year
        unique_geoid = np.unique(np.array(self.geoid))
        stratify_labels = [x[:2] for x in unique_geoid]

        idx_train, idx_val = train_test_split(unique_geoid, 
                                              test_size=0.3, stratify = stratify_labels, 
                                              random_state=self.seed, shuffle=True)

        if self.mode is not None:
            if self.mode =='train' or self.mode == 'training':
                indices = [i for i, item in enumerate(self.geoid) if any(item.startswith(prefix) for prefix in idx_train)]
                self.indices = np.array(self.matched_pair)[indices]
                self.geoid = self.geoid[indices]

            elif self.mode == 'val' or self.mode == 'validation' or self.mode == 'valid':
                indices = [i for i, item in enumerate(self.geoid) if any(item.startswith(prefix) for prefix in idx_val)]
                self.indices = np.array(self.matched_pair)[indices]
                self.geoid = self.geoid[indices]
   
        else:
            self.indices = self.matched_pair

        self.len = len(self.indices)


    def __len__(self):
        return self.len
    

    def __getitem__(self, item):
        idx = self.indices[item]
        geoid, year = idx.split('_')[0], idx.split('_')[1]
        raw_x = np.load(os.path.join(self.npy_path, '{}_{}.npy'.format(year, geoid)))

        # truncate temporal 
        raw_x = raw_x[self.start_doy_idx:self.end_doy_idx, :, :]
        y = self.labels['{}_{}'.format(geoid, year)] 

        # histogram transform
        hist_x = _calculate_histogram(raw_x, self.norm_path, num_bins=32, bands=15, channels_first=True)

        # feature selection
        x = hist_x[:, self.feature_idx, :]

        return torch.from_numpy(x).float(), torch.from_numpy(np.array(y)).float()




# npy_path = '/app/dev/spatial_encoding/data/pse'
# label_path= '/app/dev/spatial_encoding/data/composite_npy/labels.json'
# norm_path = '/app/dev/spatial_encoding/yield_prediction/data_preparation/min_max_L2_U98_hist.json'
# mode = 'train'

# dataset =  YDataset_Hist(npy_path, label_path, norm_path,
#                   lookup=[2020], mode=mode, seed=0, start_doy_idx=11, 
#                   end_doy_idx=38, feature_idx =list(range(15)))

# # Print dataset length
# print(f"Dataset length: {len(dataset)}")

# # Retrieve and print a sample
# for i in range(len(dataset)):
#     predictor, yield_data = dataset[i]
#     print(f"Sample {i}:")
#     print(f"  Predictor: {predictor.shape}")
#     print(f"  Yield: {yield_data}")
#     break
