# aiaiaik-ramba

Disclaimer: This is assuming we continue using optical imagery. If we want to try with RADAR then we have to go back a few steps... and it's not entirely easy to process SAR images.

## Goals
1. Enhance JamUNet's architecture and performance.

*MoSCoW (Must, Should, Could, Wont's)*

### Must: Reproduce and Validate JamUNet

### Should: Fix temporal error (with how the images are fed trhough at the moment)

### Should: add NDVI information (for vegetation as an indication for erosion of river channel)

### Should: Explore more indexes (e.g LAI)

### Could: get yearly elevation data (DEM)

---

## Key Steps

### 1. **Reproduce and Validate JamUNet**
   - Recreate JamUNet as described in Magherini’s thesis using the GSWD dataset.
   - Validate its performance on the Brahmaputra-Jamuna River test cases to establish a benchmark.

---

### 2. **Enhance the Model Architecture**
   - **Explore [ConvLSTMs](https://github.com/ndrplz/ConvLSTM_pytorch)**:
     - Integrate ConvLSTMs to capture spatiotemporal dynamics in sequential imagery.
     - Evaluate their performance in detecting long-term morphological changes (e.g., meander migration, erosion).
   - **Multi-Modal Inputs**:
     - Extend JamUNet to include additional inputs such as river discharge and vegetation indices (NDVI, for example).
       - River discharge is harder to estimate directly from optical imagery but can be approximated using features such as the river's lateral extent or curvature. Geologically, increased curvature often correlates with sediment deposition at bends, forming levees that can eventually redirect the river (avulsion), which indirectly reflects discharge dynamics.

---

### 3. **Adopt Transfer Learning Strategies**
Transfer learning involves leveraging models pre-trained on large datasets to accelerate training and improve performance on smaller, domain-specific datasets. 
Below are a couple of different powerful models/approaches we could try.

   - **Fine-Tune [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)**:
     - Leverage Prithvi’s pre-trained encoder for geospatial feature extraction.
     - Train a customized decoder on the GSWD dataset to predict river morphological changes.
   - **Use [TorchGeo](https://github.com/microsoft/torchgeo)**:
     - Employ TorchGeo’s pre-trained geospatial models for benchmarking.
     - Streamline data preprocessing with its specialized loaders and transformations.

---

### 4. **Experiment with Loss Functions**

Magherini used [Binary Cross Entropy (BCE)](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a), a standard loss function for binary classification tasks like distinguishing between water and non-water pixels. While effective for pixel-level accuracy, BCE doesn’t account for spatial or temporal relationships between pixels, which are crucial for understanding long-term morphological changes.

We could try:

   - **[Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)**:
     - Address outlier sensitivity in predictions, especially in areas with significant erosion or deposition changes. The Huber Loss acts like L2-loss for small errors but switches to L1-loss for larger errors (beyond some threshold), reducing impact of outliers.
   - **Multi-Task Loss**:
     - Simultaneously optimize for multiple objectives (e.g., water extent, erosion, deposition) to improve prediction accuracy, for example:
       `Total Loss = α₁ × Loss₍water₎ + α₂ × Loss₍erosion₎ + ...`
       
---

### 5. **Integrate Physics-Informed Neural Networks (PINNs)**
   - Define governing equations (e.g., sediment continuity, shallow water equations, etc).
   - Augment the loss function with physical residuals to enforce consistency with known river dynamics; for example:
     `Total Loss = Prediction Loss + λ × Physics Residual Loss`

---

