import torch
from torch.utils import data
import pandas as pd
import numpy as np
import json, os, glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from data_preparation.utils import *
from data_preparation.utils_deeplearning import *



class YDataset(data.Dataset):
    def __init__(self, npy_path, label_path, norm_path,
                  lookup=None, mode=None, seed=0, start_doy_idx=11, 
                  end_doy_idx=38, ignore_features= [9,10,11], 
                  kernel=3, feature_idx =list(range(15)), 
                  smooth = True):
        """
        
        Args:
            path (str): csv path to the main folder of the dataset
            target (str): column name of the yield column
            lookup (list): specifies list of years to filter
            seed(int): for splitting
            start_idx(int): lower bound of time series
            end_idx(int): upper bound of time series
            kernel(int): moving window size to smooth time series
            feature_idx(list): indices for subsetting at channel level
            smooth(bool): apply a convolution to smooth time series

        """
        super(YDataset, self).__init__()

        self.npy_path = npy_path
        self.label_path = label_path
        self.norm_path = norm_path
        self.lookup = lookup
        self.mode = mode
        self.seed = seed
        self.start_doy_idx = start_doy_idx
        self.end_doy_idx = end_doy_idx
        self.seq_length = self.end_doy_idx - self.start_doy_idx
        self.ignore_features=ignore_features
        self.kernel = kernel
        self.feature_idx = feature_idx
        self.smooth = smooth
 
        # prepare data for specific lookup. 15 features prepared.
        # subset at time dimension
        listX, listy, listYG  = create_data_for_years(self.lookup, self.npy_path, 
                                                self.label_path, self.norm_path, self.start_doy_idx,
                                                self.end_doy_idx, self.ignore_features, 
                                                self.smooth, self.kernel)
        
        # reshape to NxTxC
        self.X = np.stack(listX, axis=0)
        self.y = np.array(listy)
        self.geoid = np.array([x.split('_')[0] for x in listYG])
        
        assert len(listX) == len(listy) == len(self.geoid)
        
    

        if self.mode is not None:

            # # -----------------------train_test split
            # stratify on state and year
            unique_geoid = np.unique(np.array(self.geoid))
            stratify_labels = [x[:2] for x in unique_geoid]

            idx_train, idx_val = train_test_split(unique_geoid, 
                                                test_size=0.3, stratify = stratify_labels, 
                                                random_state=self.seed, shuffle=True)
            if self.mode =='train' or self.mode == 'training':
                indices = [i for i, item in enumerate(self.geoid) if any(item.startswith(prefix) for prefix in idx_train)]
                self.X = self.X[indices]
                self.y = self.y[indices]
                self.geoid = self.geoid[indices]

            elif self.mode == 'val' or self.mode == 'validation' or self.mode == 'valid':
                indices = [i for i, item in enumerate(self.geoid) if any(item.startswith(prefix) for prefix in idx_val)]
                self.X = self.X[indices]
                self.y = self.y[indices]
                self.geoid = self.geoid[indices]

        # subset at time and channel dimension
        self.X = self.X[:, :, self.feature_idx]
        self.len = self.X.shape[0]
        # print(self.X.shape, self.y.shape, len(self.geoid))
        


    def __len__(self):
        return self.len
    

    def __getitem__(self, item):
        x = self.X[item]
        y = self.y[item]

        return torch.from_numpy(x).float(), torch.from_numpy(np.array(y)).float()




# npy_path = '/app/dev/spatial_encoding/data/composite_npy'
# label_path= '/app/dev/spatial_encoding/data/composite_npy/labels.json'
# norm_path = '/app/dev/spatial_encoding/2024_08/data_preparation/min_max_L2_U98.json'

# # Initialize YieldDataset with various parameters
# dataset = YDataset(npy_path, label_path, norm_path= norm_path, 
#                   lookup=[2020], mode='validation', seed=0, start_doy_idx=11, 
#                   end_doy_idx=38, ignore_features= [9,10,11], 
#                   kernel=3, feature_idx =list(range(15)), 
#                   smooth = True)

# # Print dataset length
# print(f"Dataset length: {len(dataset)}")

# # # Retrieve and print a sample
# # for i in range(len(dataset)):
# #     predictor, yield_data = dataset[i]
# #     print(f"Sample {i}:")
# #     print(f"  Predictor: {predictor.shape}")
# #     print(f"  Yield: {yield_data}")
# #     break
