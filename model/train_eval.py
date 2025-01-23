
import torch
import torch.nn as nn
import numpy as np

from postprocessing.metrics import compute_metrics

def get_predictions(model, input_dataset, device='cuda:0'):
    '''
    Compute the predictions given the deep-learning model class and input dataset

    Inputs:
           model = class, deep-learning model
           input_dataset = TensorDataset, inputs for the model 
                           The input dataset has shape (n_samples, n_years_input=4, height=1000, width=500), 
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
                    (see the default code at the beginning of this module or in one of the training notebooks) 
    
    Output:
           predictions = list, model predictions
    '''    
    predictions = model(input_dataset.to(device))  
    return predictions.squeeze(1) 

def training_unet(model, loader, optimizer, nonwater=0, water=1, pixel_size=60, water_threshold=0.5, 
                  device='cuda:0', loss_f='BCE'):
    '''
    Training loop for the deep-learning model. Allows to choose the loss function for binary classification.
    Enables the inclusion of physics-induced loss terms (regression losses).

    Inputs: 
           model = class, deep-learning model to be trained
           loader = torch.utils.data.DataLoader element, data loader that combines a dataset 
                    and a sampler to feed data to the model in batches
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2 
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           water_threshold = float, generate binary predictions to compute the total areas of erosion and deposition
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
                    (see the default code at the beginning of this module or in one of the training notebooks)
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           loss_er_dep = str, regression loss function for erosion and deposition terms
                         default: 'Huber'. Other available options: 'RMSE', 'MAE'
                         If other loss functions are set it raises an Exception.
    
    Output: 
           losses = array of scalars, training losses 
    '''
    model.to(device)
    model.train() # specifies the model is in training mode

    losses = []
    # split in batches
    for batch in loader:
        input = batch[0].to(device)
        target = batch[1].to(device)
        target = target.float()  # Convert target tensor to float


        # get predictions
        predictions = get_predictions(model, input, device=device)
        
        # compute binary classification loss
        binary_loss = nn.BCELoss()(predictions, target)
        
        total_loss = binary_loss
        losses.append(total_loss.cpu().detach())

        # backpropagate and update weights
        optimizer.zero_grad(set_to_none=True)   # reset the computed gradients
        total_loss.backward()                   # compute the gradients using backpropagation
        optimizer.step()                        # update the weights with the optimizer
        
    losses = np.array(losses).mean() 

    return losses

def validation_unet(model, loader, nonwater=0, water=1, device='cuda:0', loss_f='BCE', water_threshold=0.5):
    '''
    Validation loop for the deep-learning model. Allows to choose the loss function for binary classification.
    Computes binary validation metrics by setting a threshold for water classification. 
    Physics-induced loss terms are not included.

    Inputs: 
           model = class, deep-learning model to be validated/tested
           loader = torch.utils.data.DataLoader element, data loader that combines a dataset 
                    and a sampler to feed data to the model in batches
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2 
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
                    (see the default code at the beginning of this module or in one of the training notebooks)
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           water_threshold = float, generate binary predictions to compute the binary validation metrics
    
    Output: 
           losses, accuracies, precisions, recalls, f1_scores, csi_scores 
                   = array of scalars, validation losses and metrics 
    '''
    model.to(device)
    model.eval() # specifies the model is in evaluation mode = validation/testing

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    csi_scores = []
    
    with torch.no_grad(): # specifies gradient is not computed
        # split in batches
        for batch in loader:
            input = batch[0].to(device)
            target = batch[1].to(device)
    
            # get predictions
            predictions = get_predictions(model, input, device=device)
            # generate binary predictions
            binary_predictions = (predictions >= water_threshold).float()
            # compute loss 
            loss = nn.BCELoss()(predictions, target)
            # compute metrics
            accuracy, precision, recall, f1_score, csi_score = compute_metrics(binary_predictions, target, nonwater, water)
            
            losses.append(loss.cpu().detach())
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            csi_scores.append(csi_score)

    losses = np.array(losses).mean()             
    accuracies = np.array(accuracies).mean()     
    precisions = np.array(precisions).mean()    
    recalls = np.array(recalls).mean()           
    f1_scores = np.array(f1_scores).mean()      
    csi_scores = np.array(csi_scores).mean()    

    return losses, accuracies, precisions, recalls, f1_scores, csi_scores

def get_erosion_deposition(previous_year_img, current_year_img, nonwater=0, water=1, pixel_size=60):
    '''
    Compute the total areas of erosion and deposition. 
    Makes a pixel-wise comparison of the last input year and the target year image. 
    The area of erosion is the sum of 'water' pixels of the target year image 
    that were 'non-water' in the last input year image. 
    The area of deposition is the sum of 'non-water' pixels of the target year image
    that were 'water' in the last input year image.

    The total number of pixels of both the areas of erosion and deposition are multiplied 
    by the pixel area (the square of `pixel_size`).

    Inputs:
           previous_year_img = 2D array or tensor, representing previous year image
           current_year_img = 2D array or tensor, representing current year image
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
    
    Output:
           list, contains total areas of erosion and deposition in km^2
    '''  
    # sum erosion and deposition pixels
    erosion = torch.sum((previous_year_img == nonwater) & (current_year_img == water))
    deposition = torch.sum((previous_year_img == water) & (current_year_img == nonwater))
    
    # calculate areas of erosion and deposition
    factor = (pixel_size**2) / (1000**2) # conversion factor to get pixel area in km^2
    erosion_areas = erosion * factor
    deposition_areas = deposition * factor
    # .item() is required to correctly save the float number ### check
    return [erosion_areas.item(), deposition_areas.item()] 