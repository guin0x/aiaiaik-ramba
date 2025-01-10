import torch

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