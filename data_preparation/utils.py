import json, os, glob
import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import pickle as pkl
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely import wkt
import matplotlib as mpl




def sits_scale_smooth(arr, ignore_idx=[9,10,11], kernel=5):

    """
    smooth specific indices in 2d array.

    ignore_idx: indices of features to ignore
    kernel: look window

    returns smoothed array
    """

    def moving_average(x, w):
        # same : returns equal sequence length
        return np.convolve(x, np.ones(w), 'same') / w

    smooth_container = []

    for i in list(range(arr.shape[1])):

        # ignore smoothing for features specified in ignore_idx
        if i in ignore_idx:
            smooth_container.append(arr[:,i])

        else:
            smooth_container.append(moving_average(arr[:, i], kernel))

    new_arr = np.moveaxis(np.stack(smooth_container), 0, -1)

    return new_arr



def normalize_channel(data, min_val, max_val):
    """Normalize the data for a single channel using min and max values."""
    data = (data - min_val) / (max_val - min_val)
    return data

def normalize_data(data, min_max_dict):
    """Normalize the data array based on min and max values provided in the dictionary."""
    normalized_data = np.empty_like(data, dtype=np.float32)  # Create an empty array for the normalized data
    

    # Normalize each channel
    for channel in range(data.shape[1]):
        min_val, max_val = min_max_dict[str(channel)]
        normalized_data[:, channel] = normalize_channel(data[:, channel], min_val, max_val)
    return normalized_data




def create_data_for_years(list_years, npy_path, labels_path, norm_path, start_doy_idx=11, end_doy_idx=42, \
                          ignore_features=[9, 10, 11], smooth=True, kernel_size=5):

    """
    create arrays for specified list of years (lookup years). start and end
    doy (day of year) index  is used to slice time sequences. See MODIS_dates.xlsx

    list_years : list containing lookup years to match arrays
    npy_path : folder containing numpy arrays of time series data
    norm_path : path to json file containing min and max of features
    labels_path: file containing crop yield labels for npy arrays
    start_doy_idx: start day of year of time sequence
    end_doy_idx: end index of time sequence
    ignore_features: list of features to ignore during smoothing
    smooth: if True, smooth array by features
    kernel_size: size of kernel for moving averaging smoothing

    returns: listX, listy, listYG
    listX = list containing arrays meeting year lookup
    listy = list containing corresponding yield labels for listX
    listYG = list containing a concatenation of year and geo-id

    """

    # containers
    listX = []  # holds numpy arrays matched to list_years
    listy = []  # holds corresponding labels to listX
    listYG = [] # holds a combination of year and county id

    # length of series
    seq_length = end_doy_idx-start_doy_idx

    # load crop yield labels
    with open(labels_path) as file:
        labels = json.load(file)

    # load crop yield labels
    with open(norm_path) as file:
        norm_values = json.load(file)

    # read npy into list
    # npys = sorted(glob.glob(npy_path +'/**/*.npy',  recursive=True))


    for i in labels:

        # get year of crop yield label and corresping county id
        year, geoid = i.split('_')[1], i.split('_')[0]

        if int(year) in list_years:

            try:
                data = np.load(os.path.join(npy_path, '{}_{}.npy'.format(year,geoid)))

                # remove uneven series and records containing nan values. ~65 records
                # if data.shape[0] == 46 and np.isnan(data).any() == False and np.isinf(data).any() == False and np.max(data)!='inf':
                if np.isnan(data).any() == False and np.isinf(data).any() == False and np.max(data)!='inf':

                    # ----- smooth series
                    if smooth == True:
                        data = sits_scale_smooth(data, ignore_features, kernel_size)

                    # normalize series
                    norm_data = normalize_data(data.copy(), norm_values)
                    listy.append(labels[i])
                    listX.append(norm_data[start_doy_idx:end_doy_idx, :])
                    listYG.append(i)

            # catch if label exist without npy
            except Exception as e:
                # print('caught ', e)
                continue

    return listX, listy, listYG



def evalMetrics(true, predicted):
    """
    returns computes metrics given reference and predicted values
    """
    mape = np.round(mean_absolute_percentage_error(true, predicted)*100,2)
    rmse = np.round(np.sqrt(mean_squared_error(true, predicted)), 2)
    r2 = np.round(r2_score(true, predicted), 2)
    corr = np.round(pearsonr(true, predicted)[0], 2)

    return mape, rmse, r2, corr


# ==================== PREDICTION MAPS

def plot_prediction_maps(county_geom,state_geom, y_true, y_pred, geoid, \
    output_dir=None, crs='epsg:4326'):

    """
    county_geom = csv containing county geometries and geoid
    state_geom = csv containing state geometries and geoid
    y_true = pickle file (.pkl) containing ground truth
    y_pred = pickle file (.pkl) containing model predictions
    geoid = pickle file (.pkl) containing geoid of prediction unit
    output_dir = directory to save maps
    crs =  coordinate system for plotting. same as geom crs
    """

    county_geom = pd.read_csv(county_geom, dtype=str)
    state_geom = pd.read_csv(state_geom, dtype=str)

    # geometry from wkt
    county_geom['geometry'] = county_geom['WKT'].apply(wkt.loads)
    state_geom['geometry'] = state_geom['WKT'].apply(wkt.loads)

    y_true = pkl.load(open(y_true, 'rb'))
    y_pred = pkl.load(open(y_pred, 'rb'))
    geoid = pkl.load(open(geoid, 'rb'))

    df = pd.DataFrame({'GEOID': pd.Series(dtype='str'),
                   'y_true': pd.Series(dtype='float'),
                   'y_pred': pd.Series(dtype='float')})

    df['GEOID'] = geoid
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['difference'] = df['y_true'] - df['y_pred']


    # colour ramp
    cmap = mpl.cm.viridis

    # pre-defined bounds
    bounds=[80, 100, 120, 140, 160, 180, 200, 220, 240]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # merge geometries table and prediction table
    merged_df  = county_geom.merge(df, on='GEOID', how='left')

    # convert to geodataframe
    merged_df = gpd.GeoDataFrame(merged_df, crs = crs)
    state_geom =  gpd.GeoDataFrame(state_geom, crs = crs)


    # plot ground/true yield values
    f, axes = plt.subplots(figsize=(20, 10), ncols=3, nrows=1, gridspec_kw = {'wspace':0, 'hspace':0})
    merged_df.plot(ax=axes[0], column='y_true', cmap=cmap, norm=norm,  missing_kwds={
                    "color": "lightgrey", "label": "Missing values"}, legend=True,
                    legend_kwds={'location':'left', 'shrink': 0.3})

    state_geom.boundary.plot(ax=axes[0], color=None, edgecolor='black', linewidth=1)

    # plot predicted yield values
    merged_df.plot(ax=axes[1], column='y_pred', cmap=cmap, norm=norm,  missing_kwds={
                    "color": "lightgrey", "label": "Missing values"}, legend=True,
                    legend_kwds={'location':'left','shrink': 0.3})
    state_geom.boundary.plot(ax=axes[1], color=None, edgecolor='black', linewidth=1)


    # plot difference between true and predicted
    merged_df.plot(ax=axes[2], column='difference', cmap='coolwarm',  missing_kwds={
                    "color": "lightgrey", "label": "Missing values"}, legend=True,
                    legend_kwds={'location':'left','shrink': 0.3})

    state_geom.boundary.plot(ax=axes[2], color=None, edgecolor='black', linewidth=1)


    axes[0].axis('off')
    axes[0].set_title('True Yield (bu/acre)')
    axes[1].axis('off')
    axes[1].set_title('Predicted Yield (bu/acre)')
    axes[2].axis('off')
    axes[2].set_title('Difference (bu/acre)')

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'prediction_map.png'))


    ## histogram
def _calculate_histogram(imagecol, norm_path, num_bins=32, bands=9, channels_first=True):
    
    """
    Given an image collection, turn it into a histogram.
    Parameters
    ----------
    imcol: The image collection to be histogrammed
    norm_path: json file, path to min,max json per feature
    num_bins: int, default=32
        The number of bins to use in the histogram.
    bands: int, default=9
        The number of bands per image. Default taken from the number of bands in the
        MOD09A1 + the number of bands in the MYD11A2 datasets
    max_bin_val: int, default=4999
        The maximum value of the bins. The default is taken from the original repository;
        note that the maximum pixel values from the MODIS datsets range from 16000 to
        18000 depending on the band
    Returns
    ----------
    A histogram for each band, of the band's pixel values. The output shape is
    [num_bins, times, bands], where times is the number of unique timestamps in the
    image collection.
    """
    
    # load min,max path
    with open(norm_path, 'r') as file:
           norm = json.loads(file.read())

    hist = []
    for idx, im in enumerate(np.split(imagecol, bands, axis=1)):
        imhist = []
        
        # generate bin sequence from min, max
        bin_seq = np.linspace(norm[str(idx)][0], norm[str(idx)][1], num_bins +1)
        
        for i in range(im.shape[0]):
            density, _ = np.histogram(im[i, :, :], bin_seq, density=False)
            # max() prevents divide by 0
            imhist.append(density / max(1, density.sum()))
        if channels_first:
            hist.append(np.stack(imhist))
        else:
            hist.append(np.stack(imhist, axis=1))
            
    return np.stack(hist, axis=1)