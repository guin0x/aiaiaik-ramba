# This module stores the function used to save the model results from a specific notebook and to reproduce these in other notebooks

import os
import torch
import copy
from pathlib import Path

import pandas as pd

def save_losses_metrics(train_losses, val_losses, metrics, batch_size, learning_rate, init_hid_dim, epochs, dir_output='model/losses_metrics'):
    """
    Save training and validation losses and metrics in a .csv file. 
    The file name includes only the hyperparameters that are being optimized.
    
    Inputs:
           train_losses = list or array, contains training losses
           val_losses = list or array, contains validation losses
           metrics = list of lists, each containing validation metrics (accuracy, precision, recall, f1_score, csi)
           batch_size = int, number of samples per batch
           learning_rate = float, learning rate
           init_hid_dim = int, initial hidden dimension of the model after first convolution
           dir_output = str, directory where the .csv file will be stored
                        default: 'model/losses_metrics'
    
    Output: 
           None, saves a .csv file with training and validation losses and metrics
    """
    # Ensure the output directory exists
    dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Generate a simplified file name
    file_name = f"losses_metrics_NDVI_bs{batch_size}_lr{learning_rate}_hid{init_hid_dim}_epoch{epochs}.csv"
    save_path = dir_output / file_name

    # Prepare the data for saving
    losses_metrics = {
        'Training loss': train_losses, 
        'Validation loss': val_losses, 
        'Accuracy': metrics[0], 
        'Precision': metrics[1], 
        'Recall': metrics[2], 
        'F1-score': metrics[3], 
        'CSI-score': metrics[4]
    }
    
    # Save as a CSV file
    df = pd.DataFrame(losses_metrics)
    df.to_csv(save_path, header=True, index=False)
    
    # Print confirmation
    print(f"Metrics saved at: {save_path}")
    return None

def save_model_path(model, batch_size, learning_rate, init_hid_dim, epochs, dir_output='model/models_trained'):
    """
    Save the model .pth file path from the training notebook. It is then loaded in a different notebook for testing the model.
    Includes only the parameters that were optimized in the file name.
    
    Inputs:
           model = class, trained deep-learning model to be validated/tested
           batch_size = int, number of samples per batch
           learning_rate = float, learning rate
           init_hid_dim = int, initial hidden dimension of the model after first convolution
           dir_output = str, general path where .pth file is stored
                        default: 'model/models_trained'
    
    Output:
           None, saves the model .pth path
    """
    best_model = copy.deepcopy(model)

    # Ensure the output directory exists
    dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Generate a simplified file name
    model_name = model.__class__.__name__
    file_name = f"{model_name}_NDVI_bs{batch_size}_lr{learning_rate}_hid{init_hid_dim}_epoch{epochs}.pth"
    
    # Full save path
    save_path = dir_output / file_name
    
    # Save the model
    torch.save(best_model.state_dict(), save_path)
    return None

def load_model(model, save_path, device):
    '''
    Load the trained model in a different notebook for the testing phase.

    Inputs:
           model = class, trained deep-learning model to be validated/tested
           save_path = str, path where the model .pth file is stored
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu'
    '''
    # load model 
    return model.load_state_dict(torch.load(save_path, map_location = torch.device(device)))