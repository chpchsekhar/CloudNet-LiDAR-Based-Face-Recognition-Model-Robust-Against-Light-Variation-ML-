# CloudNet-LiDAR-Based-Face-Recognition-Model-Robust-Against-Light-Variation-ML-
Designed a multi-modal 3D face recognition model using RGB, depth, and LiDAR point cloud data.  
# 3D Face Recognition with Spoofing Detection

## Project Overview
This project implements a 3D face recognition system with spoofing detection capabilities using:
- PyTorch for the deep learning model (CloudNet)
- Open3D for point cloud processing
- OpenCV for image processing
- PyQt5 for the graphical user interface

The system can detect four types of inputs:
1. Real faces
2. Mask attacks
3. Paper attacks
4. Replay attacks

## Key Features
- **Multimodal Input Processing**: Handles RGB images, depth maps, and 3D point clouds
- **Attention-based CNN**: Uses channel and spatial attention mechanisms
- **Interactive GUI**: Provides visual feedback and detailed results
- **Cross-platform**: Works on Windows, Linux, and macOS

## File Structure
```
.
├── model.py            # CloudNet model definition
├── train_and_save.py   # Model training and saving script
├── gui.py              # Main GUI application
├── gui1.py             # (Secondary GUI file - purpose unclear)
├── cloudnet_classification.h5  # Trained model weights (HDF5 format)
├── cloudnet_lidar_model.pth    # Trained model weights (PyTorch format)
├── notebook1_of model training.ipynb  # Model training notebook
├── notebook2_of model trained.ipynb   # Model evaluation notebook
└── README.md           # This documentation file
```

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install torch opencv-python open3d pyqt5 matplotlib numpy h5py
```

## Usage
1. Run the GUI application:
```bash
python gui.py
```
2. In the GUI:
   - Load a trained model (.h5 or .pth file)
   - Select RGB image, depth map, and point cloud files
   - Click "Process Data" to run the recognition
3. View results including:
   - Input previews
   - Classification prediction
   - Class probabilities
   - Multimodal data visualization

## Model Architecture
The CloudNet model features:
- 4 convolutional blocks with batch normalization
- Channel attention module
- Spatial attention module
- Adaptive average pooling
- 2 fully connected layers with dropout

## Training
Model training can be performed using:
- The Jupyter notebooks (`notebook1_of model training.ipynb` and `notebook2_of model trained.ipynb`)
- The `train_and_save.py` script

## License
This project is available for research and educational purposes. For commercial use, please contact the author.
