import torchnet as tnt
import json, os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import torch.optim as optim
from data_preparation.dataset_statewise import YDataset
import torch
import pickle as pkl
from data_preparation.utils import evalMetrics
from data_preparation.utils_deeplearning import *
from torch.nn import MSELoss
from models.models_1d import *
from models.ltae import LTAERegressor
from models.ms_resnet import MSResNet
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config")

def main(cfg: DictConfig):

    # create results dir and save config
    cfg.model_config.res_dir = cfg.model_config.res_dir + '_{}'.format(cfg.training.seed)
    # print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.model_config.res_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.model_config.res_dir, 'config.yaml'))

    # set seed for reproducability
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)


     # Initialize YieldDataset with various parameters
    train_dataset = YDataset(cfg.model_config.npy_path, cfg.dataset.label_path, 
                             norm_path= cfg.dataset.norm_path, lookup=cfg.dataset.train_years, 
                             mode='train', seed=cfg.training.seed, start_doy_idx=cfg.dataset.start_doy_idx, 
                             end_doy_idx=cfg.dataset.end_doy_idx, ignore_features= [9,10,11], 
                             kernel=5, feature_idx =cfg.dataset.feature_idx, 
                             smooth = True)

    # Initialize YieldDataset with various parameters
    val_dataset = YDataset(cfg.model_config.npy_path, cfg.dataset.label_path, 
                             norm_path= cfg.dataset.norm_path, lookup=cfg.dataset.train_years, 
                             mode='validation', seed=cfg.training.seed, start_doy_idx=cfg.dataset.start_doy_idx, 
                             end_doy_idx=cfg.dataset.end_doy_idx, ignore_features= [9,10,11], 
                             kernel=5, feature_idx =cfg.dataset.feature_idx, 
                             smooth = True)

    # Initialize YieldDataset with various parameters
    test_dataset_d = YDataset(cfg.model_config.npy_path, cfg.dataset.label_path, 
                             norm_path= cfg.dataset.norm_path, lookup=cfg.dataset.test_years_d, 
                             mode=None, seed=cfg.training.seed, start_doy_idx=cfg.dataset.start_doy_idx, 
                             end_doy_idx=cfg.dataset.end_doy_idx, ignore_features= [9,10,11], 
                             kernel=5, feature_idx =cfg.dataset.feature_idx, 
                             smooth = True)

    test_dataset_nd = YDataset(cfg.model_config.npy_path, cfg.dataset.label_path, 
                             norm_path= cfg.dataset.norm_path, lookup=cfg.dataset.test_years_nd, 
                             mode=None, seed=cfg.training.seed, start_doy_idx=cfg.dataset.start_doy_idx, 
                             end_doy_idx=cfg.dataset.end_doy_idx, ignore_features= [9,10,11], 
                             kernel=5, feature_idx =cfg.dataset.feature_idx, 
                             smooth = True)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=cfg.training.num_workers,  \
                                                batch_size=cfg.model_config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=cfg.training.num_workers, \
                                                batch_size=cfg.model_config.batch_size, shuffle=True )
    test_loader_d = torch.utils.data.DataLoader(test_dataset_d, num_workers=cfg.training.num_workers, \
                                                batch_size=cfg.model_config.batch_size, shuffle=False )   
    test_loader_nd = torch.utils.data.DataLoader(test_dataset_nd, num_workers=cfg.training.num_workers, \
                                                batch_size=cfg.model_config.batch_size, shuffle=False )   



    
    print('Train {}, Val {}, Test {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader_d), len(test_loader_nd)))

    # Initialize the model based on the name
    if cfg.model_config.name == "mlp":
        model = MLP(input_dim=cfg.dataset.input_dim, sequence_len =cfg.dataset.seq_length, 
                    hidden_dim1=cfg.model_config.hidden_dim1, hidden_dim2 =cfg.model_config.hidden_dim2, 
                    dropout = cfg.model_config.dropout)
        
    elif cfg.model_config.name == "lstm":
        model = LSTM(input_dim=cfg.dataset.input_dim, num_classes=1, 
                     hidden_dims=cfg.model_config.hidden_dim, num_layers=cfg.model_config.num_layers, 
                     dropout=cfg.model_config.dropout, bidirectional=cfg.model_config.bidirectional, 
                     use_layernorm=True)
        
    elif cfg.model_config.name == "lstm_attn":
        model = LSTM_Attn_mod(input_dim=cfg.dataset.input_dim, num_classes=1, 
                              hidden_dims=cfg.model_config.hidden_dim, 
                              num_layers=cfg.model_config.num_layers, dropout=cfg.model_config.dropout, 
                              bidirectional=cfg.model_config.bidirectional, use_layernorm=True)   
        
    elif cfg.model_config.name == "ltae":
        model = LTAERegressor(input_dim=cfg.dataset.input_dim, n_head=cfg.model_config.n_heads, 
                              d_k=cfg.model_config.d_k, n_neurons=[256,128], dropout=cfg.model_config.dropout, 
                              d_model=256,  T=1000, len_max_seq=cfg.dataset.seq_length, 
                              positions=cfg.model_config.positions, return_att=False)
        
    elif cfg.model_config.name == "tempcnn":
        model = TempCNN(input_dim=cfg.dataset.input_dim, num_classes=1, sequencelength=cfg.dataset.seq_length, 
                       kernel_size=cfg.model_config.kernel_size, hidden_dims=cfg.model_config.hidden_dim, 
                       dropout=cfg.model_config.dropout)
        
    elif cfg.model_config.name == "msresnet":
        model = MSResNet(input_dim=cfg.dataset.input_dim, layers=[1, 1, 1, 1], 
                         num_classes=1, hidden_dims=cfg.model_config.hidden_dim,)

    else:
        raise ValueError(f"Unknown model name: {cfg.model_config.name}")

    
    model = model.to(cfg.training.device)

   # Create optimizer from Hydra config
    optimizer_class = getattr(optim, cfg.training.optimizer.capitalize())
    optimizer = optimizer_class(model.parameters(), lr=cfg.model_config.lr)

    criterion = MSELoss()

    # Initialize TensorBoard SummaryWriter
    writer_train = SummaryWriter(log_dir=cfg.model_config.res_dir)
    writer_val = SummaryWriter(log_dir=cfg.model_config.res_dir)

    # holder for logging training performance
    trainlog = {}
    best_RMSE = np.inf
    epochs_no_improve = 0
    

    for epoch in range(1, cfg.model_config.epochs + 1):

        print('EPOCH {}/{}'.format(epoch, cfg.model_config.epochs))

        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, 
                                    device=device, display_step=cfg.training.display_step)
        # print(train_metrics)
        # Log training metrics to TensorBoard
        writer_train.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
        writer_train.add_scalar('R2/train', train_metrics['train_R2'], epoch)

        # print('Validation . . . ')
        model.eval()
        val_metrics = evaluation(model, criterion, val_loader, device=device, mode='val')
        print('Loss {:.4f},  RMSE {:.4f}, R2 {:.4f}'.format(val_metrics['val_loss'], 
                                                            val_metrics['val_rmse'], 
                                                            val_metrics['val_R2']))

        # Log validation metrics to TensorBoard
        writer_val.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        # writer_val.add_scalar('RMSE/val', val_metrics['val_rmse'], epoch)
        writer_val.add_scalar('R2/val', val_metrics['val_R2'], epoch)

        trainlog[epoch] = {**train_metrics, **val_metrics}
        checkpoint(trainlog, cfg.model_config.res_dir)
        

        # Early stopping
        if val_metrics['val_rmse'] < best_RMSE:
            best_epoch = epoch
            best_RMSE = val_metrics['val_rmse']
            epochs_no_improve = 0  # Reset the counter if validation loss improves
            torch.save({'best epoch': best_epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(cfg.model_config.res_dir, 'model.pth.tar'))
        else:
            epochs_no_improve += 1  # Increment the counter if validation loss does not improve
        
        if epochs_no_improve >= cfg.model_config.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break  # Stop training if no improvement for specified number of epochs


    # load best model
    model.load_state_dict(torch.load(os.path.join(cfg.model_config.res_dir, 'model.pth.tar'))['state_dict'])

    # evaluate on test data
    model.eval()
    test_metrics, y_true, y_pred = evaluation(model, criterion, test_loader_d, device=device, mode='test')
    print('========== Test Metrics ===========')
    print('Loss {:.4f},  RMSE {:.4f}, R2 {:.4f}'.format(test_metrics['test_loss'], 
                                                        test_metrics['test_rmse'], 
                                                        test_metrics['test_R2']))
    save_results(test_metrics, cfg.model_config.res_dir, y_true, y_pred, test_dataset_d.geoid ,cfg.dataset.test_years_d[0])

    # evaluate on test data
    model.eval()
    test_metrics, y_true, y_pred = evaluation(model, criterion, test_loader_nd, device=device, mode='test')
    print('========== Test Metrics ===========')
    print('Loss {:.4f},  RMSE {:.4f}, R2 {:.4f}'.format(test_metrics['test_loss'], 
                                                        test_metrics['test_rmse'], 
                                                        test_metrics['test_R2']))
    save_results(test_metrics, cfg.model_config.res_dir, y_true, y_pred, test_dataset_nd.geoid, cfg.dataset.test_years_nd[0])



    # close the TensorBoard writer
    writer_train.close()
    writer_val.close()

if __name__ == "__main__":
    main()