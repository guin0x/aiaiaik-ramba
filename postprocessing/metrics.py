import torch
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                            average_precision_score, balanced_accuracy_score)
import matplotlib.pyplot as plt

def compute_metrics(pred, target, nonwater_value=0, water_value=1, water_threshold=0.5):
    '''
    Compute the metrics (accuracy, recall, precision, F1 score and CSI score) based on assigned water threshold.
    Values larger than the threshold are set equal to 1, whereas lower values are set to 0.
    Positive = water (1). Negative = non-water (0)

    Inputs:
           pred = 2D array or tensor, model predictions
           target =  2D array or tensor, real targets
           nonwater_value = int, represents pixel value of non-water class.
                         default: 0, based on the updated pixel classes. 
           water_value = int, represents pixel value of water class.
                         default: 1, based on the updated pixel classes. 
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)

    Output:
           accuracy, precision, recall, f1_score, csi = floats, classification metrics 
    '''
    # generate binary predictions
    binary_predictions = (pred >= water_threshold).float()
    
    # false negative is predicted non-water but actually water and the other way around for false positive
    tp = torch.sum((binary_predictions == water_value) & (target == water_value)).item()
    fp = torch.sum((binary_predictions == water_value) & (target == nonwater_value)).item()
    tn = torch.sum((binary_predictions == nonwater_value) & (target == nonwater_value)).item()
    fn = torch.sum((binary_predictions == nonwater_value) & (target == water_value)).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return accuracy, precision, recall, f1_score, csi 

def single_roc_curve(model, dataset, sample, train_val_test='testing', device='cuda:0', get_avg=False):
    '''
    Plot or return the FPR, TPR and ROC AUC of a single sample.

    Inputs:
           model = class, deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           sample = int, specifies the input-target combination
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           get_avg = bool, sets whether the function plots ROC curve or only returns fpr, tpr and roc_auc 
                     default: False, plots the curve
    Output:
           None or fpr, tpr, roc_auc, depending on get_avg key
    '''
    # need internal import statement to avoid circular imports
    from model.train_eval import get_predictions

    single_input = dataset[sample][0]
    single_target = dataset[sample][1].cpu()
    
    target_flat = single_target.flatten()

    prediction = get_predictions(model, single_input.unsqueeze(0), device)
    prediction_flat = prediction.detach().cpu().flatten() # unsqueeze needed to match dimensions
    
    # compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(target_flat, prediction_flat)
    roc_auc = auc(fpr, tpr)

    if not get_avg:
        plt.figure()
        plt.plot(fpr, tpr, color='navy', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.fill_between(fpr, tpr,  color='palegoldenrod')
        plt.plot([0, 1], [0, 1], color='red', lw=2.5, linestyle='--', label=f'Random classifier = 0.5')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate [-]', fontsize=14)
        plt.ylabel('True Positive Rate [-]', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'Receiver Operating Characteristic curve\n for {train_val_test} sample {sample}', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

    return fpr, tpr, roc_auc if get_avg else None