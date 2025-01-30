## README: `data/` Folder  

This README outlines the structure and specifics of the `data/` folder, which contains preprocessed datasets for training and evaluating deep learning models in river morphodynamics.  

### Folder Contents  

The `data/` folder contains the following files:  

1. **Non-3D Datasets** (4 input channels for four consecutive years):
   - `train_set.pkl`  
   - `val_set.pkl`  
   - `test_set.pkl`  

2. **3D Datasets** (1 input channel with time-stacked data for four consecutive years):  
   - `train_set_3d.pkl`  
   - `val_set_3d.pkl`  
   - `test_set_3d.pkl`  

### Data Description  

- **Non-3D datasets**:  
  Each sample includes four input channels corresponding to satellite images from four consecutive years. This format allows the model to analyze temporal changes in river morphology.  

- **3D datasets**:  
  These datasets stack the four yearly images into a single 3D input channel, providing a compressed temporal representation. This format is particularly useful for architectures designed to handle spatiotemporal data in a unified manner.  

### Preprocessing Pipeline  

1. **Purpose**:  
   The datasets were pickled to bypass preprocessing inconsistencies between local and RunPod environments.  

2. **Consistency Check**:  
   We ensured the integrity of the preprocessed datasets using the following assertions (for all train/test/val datasets):  
   ```python  
   assert torch.equal(train_set_prep.tensors[0], train_set_antonio.tensors[0]), "Inputs differ"  
   assert torch.equal(train_set_prep.tensors[1], train_set_antonio.tensors[1]), "Targets differ"  
   ```  
   This confirmed that the inputs and targets in the pickled files matched the original preprocessed data.  

