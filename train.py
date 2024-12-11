from torch import dropout, nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torchnet as tnt
import torch
import torch.optim as optim
from models.utils import *
from models.lstm import LSTM
from data_preparation.dataset import YieldDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path="conf", config_name="config")

def main(cfg: DictConfig):

    # print(OmegaConf.to_yaml(cfg))

    # create results dir and save config
    os.makedirs(cfg.dataset.results_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.dataset.results_path, 'config.yaml'))

    # set seed for reproducability
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)


    # Initialize YieldDataset with various parameters
    train_dataset = YieldDataset(
        predictor_path= cfg.dataset.predictor_path,
        yield_path= cfg.dataset.yield_path,
        years= cfg.dataset.train_years,
        feature_selector= cfg.dataset.feature_selector,
        max_timesteps= cfg.dataset.max_timesteps,
        temporal_truncation= cfg.dataset.temporal_truncation, 
        proportion= cfg.dataset.proportion,
        state_selector= cfg.dataset.state_selector,
        aez_selector= cfg.dataset.aez_selector
    )

    val_dataset = YieldDataset(
        predictor_path= cfg.dataset.predictor_path,
        yield_path= cfg.dataset.yield_path,
        years= cfg.dataset.val_years,
        feature_selector= cfg.dataset.feature_selector,
        max_timesteps= cfg.dataset.max_timesteps,
        temporal_truncation= cfg.dataset.temporal_truncation, 
        proportion= cfg.dataset.proportion,
        state_selector= cfg.dataset.state_selector,
        aez_selector= cfg.dataset.aez_selector
    )

    test_dataset = YieldDataset(
        predictor_path= cfg.dataset.predictor_path,
        yield_path= cfg.dataset.yield_path,
        years= cfg.dataset.test_years,
        feature_selector= cfg.dataset.feature_selector,
        max_timesteps= cfg.dataset.max_timesteps,
        temporal_truncation= cfg.dataset.temporal_truncation, 
        proportion= cfg.dataset.proportion,
        state_selector= cfg.dataset.state_selector,
        aez_selector= cfg.dataset.aez_selector
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=cfg.training.num_workers,  \
                                                batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=cfg.training.num_workers, \
                                                batch_size=cfg.training.batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=cfg.training.num_workers, \
                                                batch_size=cfg.training.batch_size, shuffle=False )   

    # get year for each sample in test_loader
    prediction_years = test_dataset.years 
    
    print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))
    
    model =  LSTM(input_dim=cfg.model.input_dim, num_classes=cfg.model.num_classes, 
                  hidden_dims=cfg.model.hidden_dims,num_layers=cfg.model.num_layers, 
                  dropout=cfg.model.dropout, bidirectional=cfg.model.bidirectional, 
                  use_layernorm=cfg.model.use_layernorm)
    
    model = model.to(cfg.training.device)

   # Create optimizer from Hydra config
    optimizer_class = getattr(optim, cfg.training.optimizer.capitalize())
    optimizer = optimizer_class(model.parameters(), lr=cfg.training.learning_rate)

    criterion = MSELoss()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=cfg.dataset.results_path)

    # holder for logging training performance
    trainlog = {}
    best_RMSE = np.inf
    epochs_no_improve = 0 
    

    for epoch in range(1, cfg.training.epochs + 1):

        print('EPOCH {}/{}'.format(epoch, cfg.training.epochs))

        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, 
                                    device=device, display_step=cfg.training.display_step)
        # print(train_metrics)
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
        writer.add_scalar('R2/train', train_metrics['train_R2'], epoch)

        # print('Validation . . . ')
        model.eval()
        val_metrics = evaluation(model, criterion, val_loader, device=device, mode='val')
        print('Loss {:.4f},  NRMSE {:.4f}, R2 {:.4f}'.format(val_metrics['val_loss'], 
                                                            val_metrics['val_nrmse'], 
                                                            val_metrics['val_R2']))

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        writer.add_scalar('NRMSE/val', val_metrics['val_nrmse'], epoch)
        writer.add_scalar('R2/val', val_metrics['val_R2'], epoch)

        trainlog[epoch] = {**train_metrics, **val_metrics}
        checkpoint(trainlog, cfg.dataset.results_path)
        

        # Early stopping
        if val_metrics['val_nrmse'] < best_RMSE:
            best_epoch = epoch
            best_RMSE = val_metrics['val_nrmse']
            epochs_no_improve = 0  # Reset the counter if validation loss improves
            torch.save({'best epoch': best_epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(cfg.dataset.results_path, 'model.pth.tar'))
        else:
            epochs_no_improve += 1  # Increment the counter if validation loss does not improve
        
        if epochs_no_improve >= cfg.training.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break  # Stop training if no improvement for specified number of epochs


    # load best model
    model.load_state_dict(torch.load(os.path.join(cfg.dataset.results_path, 'model.pth.tar'))['state_dict'])

    # evaluate on test data
    model.eval()
    test_metrics, y_true, y_pred = evaluation(model, criterion, test_loader, device=device, mode='test')
    print('========== Test Metrics ===========')
    print('Loss {:.4f},  NRMSE {:.4f}, R2 {:.4f}'.format(test_metrics['test_loss'], 
                                                        test_metrics['test_nrmse'], 
                                                        test_metrics['test_R2']))
    print('========== Test Metrics ===========')
    save_results(test_metrics, cfg.dataset.results_path, y_true, y_pred, prediction_years)

    # log test metrics to TensorBoard
    writer.add_scalar('Loss/test', test_metrics['test_loss'])
    writer.add_scalar('NRMSE/test', test_metrics['test_nrmse'])
    writer.add_scalar('R2/test', test_metrics['test_R2'])

    # close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()