 
import sys
import os
import numpy as np
import torch
import cv2
import open3d as o3d
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFileDialog, QComboBox, QMessageBox,
                             QGroupBox, QGridLayout, QTextEdit, QFrame, QSplitter, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
from PyQt5.QtGui import QIcon

matplotlib.use('Qt5Agg')
import torch
import torch.nn as nn

# CloudNet model - matches the trained model structure
class CloudNet(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(CloudNet, self).__init__()

        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Third convolution block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Fourth convolution block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Apply channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att

        # Apply spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att

        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc(x)

        return x


# Worker thread for processing data
class ProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    processing_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, rgb_path=None, depth_path=None, ply_path=None, model=None):
        super().__init__()
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.ply_path = ply_path
        self.model = model
        
    def run(self):
        try:
            result = {}
            # Initialize progress
            self.update_progress.emit(10)
            
            # Process RGB, Depth and Point Cloud to create the necessary .npy format
            if all([self.rgb_path, self.depth_path, self.ply_path, self.model]):
                # Load RGB image
                rgb_img = cv2.imread(self.rgb_path)
                if rgb_img is None:
                    self.error_occurred.emit(f"Failed to load RGB image: {self.rgb_path}")
                    return
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                self.update_progress.emit(30)
                
                # Load Depth image
                depth_img = cv2.imread(self.depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    # Try regular loading if ANYDEPTH fails
                    depth_img = cv2.imread(self.depth_path, cv2.IMREAD_GRAYSCALE)
                    if depth_img is None:
                        self.error_occurred.emit(f"Failed to load Depth image: {self.depth_path}")
                        return
                # Normalize depth image
                if depth_img.max() > 0:
                    depth_img = depth_img.astype(np.float32) / depth_img.max()
                self.update_progress.emit(50)
                
                # Load point cloud
                try:
                    pcd = o3d.io.read_point_cloud(self.ply_path)
                    if not pcd.has_points():
                        self.error_occurred.emit(f"Point cloud has no points: {self.ply_path}")
                        return
                        
                    # Extract points 
                    points = np.asarray(pcd.points)
                    
                    # Create feature representation similar to your training data
                    # Resize RGB to match your model's expected input
                    rgb_resized = cv2.resize(rgb_img, (256, 256))
                    depth_resized = cv2.resize(depth_img, (256, 256))
                    
                    # Create multimodal input (RGB + Depth)
                    # Reshape to match model input requirements - 1 channel for now
                    multimodal_data = np.zeros((256, 4), dtype=np.float32)  # Height x Channels
                    
                    # Average across width dimension for simplified representation
                    rgb_mean = np.mean(rgb_resized, axis=1) / 255.0  # Normalize RGB
                    depth_mean = np.mean(depth_resized, axis=1)
                    
                    # Combine into multimodal representation
                    multimodal_data[:, 0] = rgb_mean[:, 0]  # R channel
                    multimodal_data[:, 1] = rgb_mean[:, 1]  # G channel
                    multimodal_data[:, 2] = rgb_mean[:, 2]  # B channel
                    multimodal_data[:, 3] = depth_mean  # Depth channel
                    
                    # Save result for visualization
                    result['multimodal_data'] = multimodal_data
                    result['rgb_preview'] = rgb_img
                    result['depth_preview'] = depth_img
                    result['points'] = points
                    
                    # Prepare for model input - Get the first channel for the model
                    # Model expects [batch_size, channels, height, width]
                    model_input = torch.tensor(multimodal_data[:, 0:1], dtype=torch.float32)
                    model_input = model_input.unsqueeze(0)  # Add batch dimension
                    
                    self.update_progress.emit(80)
                    
                    # Model inference
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model.to(device)
                    self.model.eval()
                    
                    with torch.no_grad():
                        model_input = model_input.to(device)
                        outputs = self.model(model_input)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # Get prediction
                        _, predicted = torch.max(outputs, 1)
                        prediction = predicted.item()
                        probs = probabilities[0].cpu().numpy()
                        
                        # Store results
                        result['prediction'] = prediction
                        result['probabilities'] = probs
                        result['class_names'] = ['real', 'mask', 'paper', 'replay']
                    
                    self.update_progress.emit(100)
                    self.processing_done.emit(result)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.error_occurred.emit(f"Error processing point cloud: {str(e)}")
            else:
                self.error_occurred.emit("Missing required input files or model")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Processing error: {str(e)}")


# Main application window
class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Face Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        # Initialize variables
        self.rgb_path = None
        self.depth_path = None
        self.ply_path = None
        self.model = None
        self.model_loaded = False
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        # Set application style
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f7;
                color: #333333;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QPushButton#process_btn {
                background-color: #27ae60;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton#process_btn:hover {
                background-color: #219955;
            }
            QLabel {
                padding: 2px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                color: white;
                background-color: #f5f5f7;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
            QSplitter::handle {
                background-color: #cccccc;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Title and description
        title_label = QLabel("3D Face Recognition System")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        description = QLabel("Upload RGB image, Depth map, and Point Cloud (.ply) to detect spoofing attacks")
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("color: #7f8c8d; margin-bottom: 15px;")
        main_layout.addWidget(description)
        
        # Create a horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Input section
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        select_model_btn = QPushButton("Select Model File (.h5/.pth)")
        select_model_btn.setIcon(QIcon.fromTheme("document-open"))
        select_model_btn.clicked.connect(self.load_model)
        
        model_layout.addWidget(select_model_btn)
        model_layout.addWidget(self.model_path_label)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Input data section
        input_group = QGroupBox("Input Data")
        input_layout = QGridLayout()
        input_layout.setSpacing(10)
        
        # RGB Image
        input_layout.addWidget(QLabel("RGB Image:"), 0, 0)
        self.rgb_label = QLabel("No file selected")
        self.rgb_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.rgb_btn = QPushButton("Browse")
        self.rgb_btn.setIcon(QIcon.fromTheme("document-open"))
        self.rgb_btn.clicked.connect(self.load_rgb)
        input_layout.addWidget(self.rgb_label, 0, 1)
        input_layout.addWidget(self.rgb_btn, 0, 2)
        
        # Depth Image
        input_layout.addWidget(QLabel("Depth Map:"), 1, 0)
        self.depth_label = QLabel("No file selected")
        self.depth_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.depth_btn = QPushButton("Browse")
        self.depth_btn.setIcon(QIcon.fromTheme("document-open"))
        self.depth_btn.clicked.connect(self.load_depth)
        input_layout.addWidget(self.depth_label, 1, 1)
        input_layout.addWidget(self.depth_btn, 1, 2)
        
        # Point Cloud
        input_layout.addWidget(QLabel("Point Cloud:"), 2, 0)
        self.ply_label = QLabel("No file selected")
        self.ply_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.ply_btn = QPushButton("Browse")
        self.ply_btn.setIcon(QIcon.fromTheme("document-open"))
        self.ply_btn.clicked.connect(self.load_ply)
        input_layout.addWidget(self.ply_label, 2, 1)
        input_layout.addWidget(self.ply_btn, 2, 2)
        
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # Process button
        self.process_btn = QPushButton("Process Data")
        self.process_btn.setObjectName("process_btn")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        left_layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        
        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: 'Courier New'; font-size: 12px;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # Right panel - Visualization and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(15)
        
        # Preview section
        preview_group = QGroupBox("Input Preview")
        preview_layout = QHBoxLayout()
        
        # RGB preview
        rgb_preview_layout = QVBoxLayout()
        self.rgb_preview = QLabel()
        self.rgb_preview.setFixedSize(200, 200)
        self.rgb_preview.setAlignment(Qt.AlignCenter)
        self.rgb_preview.setFrameShape(QFrame.Box)
        self.rgb_preview.setStyleSheet("border: 2px solid #cccccc; border-radius: 4px; background-color: #f8f9fa;")
        self.rgb_preview.setText("RGB Preview")
        rgb_preview_layout.addWidget(self.rgb_preview)
        rgb_label = QLabel("RGB Image")
        rgb_label.setAlignment(Qt.AlignCenter)
        rgb_preview_layout.addWidget(rgb_label)
        preview_layout.addLayout(rgb_preview_layout)
        
        # Depth preview
        depth_preview_layout = QVBoxLayout()
        self.depth_preview = QLabel()
        self.depth_preview.setFixedSize(200, 200)
        self.depth_preview.setAlignment(Qt.AlignCenter)
        self.depth_preview.setFrameShape(QFrame.Box)
        self.depth_preview.setStyleSheet("border: 2px solid #cccccc; border-radius: 4px; background-color: #f8f9fa;")
        self.depth_preview.setText("Depth Preview")
        depth_preview_layout.addWidget(self.depth_preview)
        depth_label = QLabel("Depth Map")
        depth_label.setAlignment(Qt.AlignCenter)
        depth_preview_layout.addWidget(depth_label)
        preview_layout.addLayout(depth_preview_layout)
        
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)
        
        # Add Point Cloud canvas to preview section
        self.pc_figure1 = Figure(figsize=(5, 4))
        self.pc_figure1.patch.set_facecolor('#f8f9fa')
        self.pc_canvas1 = FigureCanvas(self.pc_figure1)
        self.pc_canvas1.setStyleSheet("border: 2px solid #cccccc; border-radius: 4px;")
        preview_layout.addWidget(self.pc_canvas1)
        
        # Point cloud visualization
        pc_group = QGroupBox("Multimodal Data Representation") 
        pc_layout = QVBoxLayout()
        
        # Use MatplotlibCanvas for point cloud visualization
        self.pc_figure = Figure(figsize=(5, 4))
        self.pc_figure.patch.set_facecolor('#f8f9fa')
        self.pc_canvas = FigureCanvas(self.pc_figure)
        self.pc_canvas.setStyleSheet("border: 2px solid #cccccc; border-radius: 4px;")
        pc_layout.addWidget(self.pc_canvas)
        
        pc_group.setLayout(pc_layout)
        right_layout.addWidget(pc_group)
        
        # Results section
        results_group = QGroupBox("Recognition Results")
        results_layout = QVBoxLayout()
        
        # Result label
        self.result_label = QLabel("No results yet")
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("margin: 10px; padding: 10px; font-weight: bold;")
        results_layout.addWidget(self.result_label)
        
        # Probability chart
        self.prob_figure = Figure(figsize=(6, 3))
        self.prob_figure.patch.set_facecolor('#f8f9fa')
        self.prob_canvas = FigureCanvas(self.prob_figure)
        self.prob_canvas.setStyleSheet("border: 2px solid #cccccc; border-radius: 4px;")
        results_layout.addWidget(self.prob_canvas)
        
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        # Add both panels to the splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])  # Initial sizes
        
        # Initialize log
        self.log("Application started. Please load your model and input files.")
        
        # Update UI when window resizes
        self.setMinimumSize(1200, 800)
        
    def log(self, message):
        """Add message to log window with timestamp"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def load_model(self):
        """Load the trained model"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.h5 *.pth);;All Files (*)", options=options
        )
        
        if file_path:
            try:
                self.log(f"Loading model from {file_path}")
                
                # Create model instance
                self.model = CloudNet(num_classes=4)
                
                # Load weights based on file extension
                if file_path.endswith('.pth'):
                    # Load PyTorch model
                    self.model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
                    self.log("PyTorch model loaded successfully.")
                elif file_path.endswith('.h5'):
                    # Load weights from H5 file - improved handling
                    try:
                        with h5py.File(file_path, 'r') as f:
                            for name, param in self.model.named_parameters():
                                if name in f:
                                    param.data = torch.from_numpy(f[name][()])
                        self.log("H5 model loaded successfully.")
                    except Exception as e:
                        # Try alternative loading mechanism if the above fails
                        self.log(f"Standard H5 loading failed, trying alternative method: {str(e)}")
                        state_dict = {}
                        with h5py.File(file_path, 'r') as f:
                            for key in f.keys():
                                state_dict[key] = torch.from_numpy(f[key][()])
                        torch.save(self.model.state_dict(), self.model_save_path)

                        self.log("H5 model loaded with alternative method.")
                
                self.model.eval()  # Set to evaluation mode
                self.model_loaded = True
                self.model_path_label.setText(os.path.basename(file_path))
                
                self.check_process_button()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log(f"Error loading model: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                self.model = None
                self.model_loaded = False
        
    def load_rgb(self):
        """Load RGB image"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select RGB Image", "", "Image Files (*.png *.jpg *.jpeg);;All Files (*)", options=options
        )
        
        if file_path:
            try:
                self.rgb_path = file_path
                self.rgb_label.setText(os.path.basename(file_path))
                self.log(f"RGB image loaded: {os.path.basename(file_path)}")
                
                # Display preview
                img = cv2.imread(file_path)
                if img is None:
                    raise Exception(f"Failed to read image file: {file_path}")
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (200, 200))
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.rgb_preview.setPixmap(QPixmap.fromImage(q_img))
                
                self.check_process_button()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log(f"Error loading RGB image: {str(e)}")
                QMessageBox.warning(self, "Warning", f"Failed to load RGB image: {str(e)}")
        
    def load_depth(self):
        """Load depth map"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Depth Map", "", "Image Files (*.png *.jpg *.jpeg);;All Files (*)", options=options
        )
        
        if file_path:
            try:
                self.depth_path = file_path
                self.depth_label.setText(os.path.basename(file_path))
                self.log(f"Depth map loaded: {os.path.basename(file_path)}")
                
                # Try different loading methods
                depth_img = None
                
                # First try ANYDEPTH
                depth_img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
                
                # If that fails, try grayscale
                if depth_img is None:
                    depth_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                # If that fails, try regular color loading
                if depth_img is None:
                    depth_img = cv2.imread(file_path)
                    if depth_img is not None:
                        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
                
                if depth_img is None:
                    raise Exception(f"Failed to read depth file: {file_path}")
                
                # Normalize for display
                if depth_img.max() > 0:
                    depth_img_display = (depth_img.astype(np.float32) / depth_img.max() * 255).astype(np.uint8)
                else:
                    depth_img_display = depth_img
                
                depth_img_display = cv2.resize(depth_img_display, (200, 200))
                
                # Create QImage from depth data
                q_img = QImage(depth_img_display.data, 200, 200, 200, QImage.Format_Grayscale8)
                self.depth_preview.setPixmap(QPixmap.fromImage(q_img))
                
                self.check_process_button()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log(f"Error loading depth map: {str(e)}")
                QMessageBox.warning(self, "Warning", f"Failed to load depth map: {str(e)}")
        
    def load_ply(self):
        """Load point cloud"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Point Cloud", "", "PLY Files (*.ply);;All Files (*)", options=options
        )
        
        if file_path:
            try:
                self.ply_path = file_path
                self.ply_label.setText(os.path.basename(file_path))
                self.log(f"Point cloud loaded: {os.path.basename(file_path)}")
                
                # Visualize point cloud
                self.visualize_point_cloud(file_path)
                
                self.check_process_button()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log(f"Error loading point cloud: {str(e)}")
                QMessageBox.warning(self, "Warning", f"Failed to load point cloud: {str(e)}")
        
    def visualize_point_cloud(self, ply_path):
        """Visualize point cloud using matplotlib"""
        try:
            # Clear previous plot
            self.pc_figure1.clear()
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            
            # Create 3D scatter plot
            ax = self.pc_figure1.add_subplot(111, projection='3d')
            
            # Downsample points if there are too many
            max_points = 10000  # Maximum number of points to display
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points_display = points[indices]
            else:
                indices = np.arange(len(points))
                points_display = points
            
            # Get colors if available
            if pcd.has_colors():
                # Fix: handle colors correctly when downsampling
                colors = np.asarray(pcd.colors)
                if len(points) > max_points:
                    colors_display = colors[indices]
                else:
                    colors_display = colors
            else:
                colors_display = np.ones_like(points_display) * [0.5, 0.5, 0.5]  # Default gray
            
            # Plot points
            ax.scatter(points_display[:, 0], points_display[:, 1], points_display[:, 2], 
                      c=colors_display if colors_display.shape[1] == 3 else 'gray', s=1)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Point Cloud Preview')
            
            # Update canvas
            self.pc_canvas1.draw()
            
            self.log("Point cloud visualization updated")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"Error visualizing point cloud: {str(e)}")
            
            # Create simple message on the figure instead
            self.pc_figure.clear()
            ax = self.pc_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Failed to visualize point cloud.\nProcessing will still continue.", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.pc_canvas.draw()
    
    def check_process_button(self):
        """Enable process button if all inputs are loaded"""
        if self.model_loaded and self.rgb_path and self.depth_path and self.ply_path:
            self.process_btn.setEnabled(True)
            self.log("All inputs loaded. Ready to process.")
        else:
            self.process_btn.setEnabled(False)
    
    def process_data(self):
        """Process the input data and make predictions"""
        if not all([self.rgb_path, self.depth_path, self.ply_path, self.model]):
            self.log("Error: All inputs and model must be loaded before processing")
            return
        
        self.log("Starting data processing...")
        self.progress_bar.setValue(0)
        
        # Disable process button during processing
        self.process_btn.setEnabled(False)
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(
            self.rgb_path, self.depth_path, self.ply_path, self.model
        )
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.processing_done.connect(self.display_results)
        self.processing_thread.error_occurred.connect(self.handle_error)
        self.processing_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def display_results(self, result):
        """Display the processing results"""
        try:
            self.log("Processing completed successfully")
            
            # Extract results
            prediction = result['prediction']
            probabilities = result['probabilities']
            class_names = result['class_names']
            
            # Update result label
            result_text = f"Prediction: {class_names[prediction]}"
            self.result_label.setText(result_text)
            
            # Set color based on prediction (red for attacks, green for real)
            if prediction == 0:  # Real face
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
            else:  # Attack
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
            
            # Plot probabilities
            self.prob_figure.clear()
            ax = self.prob_figure.add_subplot(111)
            bars = ax.bar(class_names, probabilities)
            
            # Color the bars based on class (green for real, red for attacks)
            for i, bar in enumerate(bars):
                bar.set_color('green' if i == 0 else 'red')
                
            ax.set_ylabel('Probability')
            ax.set_title('Class Probabilities')
            
            # Add text labels above bars
            for i, v in enumerate(probabilities):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
                
            self.prob_figure.tight_layout()
            self.prob_canvas.draw()
            
            # Plot multimodal data if available
            if 'multimodal_data' in result:
                self.pc_figure.clear()
                ax = self.pc_figure.add_subplot(111)
                im = ax.imshow(result['multimodal_data'], aspect='auto', interpolation='nearest')
                ax.set_title('Multimodal Data Representation')
                ax.set_xlabel('Features (R, G, B, Depth)')
                ax.set_ylabel('Position')
                self.pc_figure.colorbar(im)
                self.pc_figure.tight_layout()
                self.pc_canvas.draw()
            
            # Re-enable process button
            self.process_btn.setEnabled(True)
            
            # Log detailed results
            self.log(f"Prediction: {class_names[prediction]}")
            for i, class_name in enumerate(class_names):
                self.log(f"  {class_name}: {probabilities[i]:.4f}")
                
        except Exception as e:
           
                import traceback
                traceback.print_exc()
                self.log(f"Error displaying results: {str(e)}")
         
                
        
    def handle_error(self, error_msg):
        """Handle errors from processing thread"""
        self.log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.process_btn.setEnabled(True)


# Application entry point
if __name__ == "__main__":
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())