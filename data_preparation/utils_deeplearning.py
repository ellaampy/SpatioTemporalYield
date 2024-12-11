import torch
from torch import dropout, nn
import torch.utils.data as data
from torch.nn import MSELoss
import torch.nn.functional as F
import torchnet as tnt
import os, json
import pickle as pkl
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score







# ====================== TRAIN AND EVAL ITERATOR
def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def train_epoch(model, optimizer, criterion, data_loader, device, display_step):
    rmse_meter = tnt.meter.MSEMeter(root=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(data_loader):
        
        y_true.extend(list(map(float, y)))

        x = recursive_todevice(x, device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        rmse_meter.add(out, y)
        loss_meter.add(loss.item())

        pred = out.detach().cpu().numpy()
        y_pred.extend(list(pred))


        if (i + 1) % display_step == 0:
            print('Step [{}/{}], Loss: {:.4f}, RMSE : {:.2f}'.format(
                i + 1, len(data_loader), loss_meter.value()[0], rmse_meter.value()))

    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_rmse': rmse_meter.value(),
                     'train_R2': r2_score(np.array(y_true), np.array(y_pred)),
                     'train_r': pearsonr(np.array(y_true), np.array(y_pred))[0],
                     'train_mape': mean_absolute_percentage_error(np.array(y_true), np.array(y_pred))}

    return epoch_metrics



def evaluation(model, criterion, loader, device, mode='val'):
    y_true = []
    y_pred = []

    rmse_meter = tnt.meter.MSEMeter(root=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, y) in loader:

        y_true.extend(list(map(float, y)))
        x = recursive_todevice(x, device)
        # x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
            loss = criterion(prediction, y)

        rmse_meter.add(prediction, y)
        loss_meter.add(loss.item())

        pred = prediction.cpu().numpy()
        y_pred.extend(list(pred))

    metrics = {'{}_rmse'.format(mode): rmse_meter.value(),
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_R2'.format(mode): r2_score(np.array(y_true), np.array(y_pred)), 
               '{}_r'.format(mode): pearsonr(np.array(y_true), np.array(y_pred))[0],
               '{}_mape'.format(mode): mean_absolute_percentage_error(np.array(y_true), np.array(y_pred))}

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics, y_true, y_pred 



# ================ HELPER FUNCTIONS I/O
def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]
    
    

def prepare_output(res_dir):
    os.makedirs(res_dir, exist_ok=True)


    
def checkpoint(log, res_dir):
    with open(os.path.join(res_dir, 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, res_dir, y_true, y_pred, geoid, year):

    # modified - year added
    with open(os.path.join(res_dir,'test_metrics_{}.json'.format(year)), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
        
    # save y_true, y_pred, geoid as pickle
    pkl.dump(y_true, open(os.path.join(res_dir, 'y_true_test_data_{}.pkl'.format(year)), 'wb'))
    pkl.dump(y_pred, open(os.path.join(res_dir,  'y_pred_test_data_{}.pkl'.format(year)), 'wb'))
    pkl.dump(geoid.tolist(), open(os.path.join(res_dir,  'geoid_{}.pkl'.format(year)), 'wb'))

    
    

# =====================