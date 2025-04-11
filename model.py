import torch
import torch.nn as nn
import torch.nn.functional as F

class CloudNet(nn.Module):
    def __init__(self):
        super(CloudNet, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Automatically determine the correct input size for fc1
        self.flattened_size = self._get_flattened_size()

        # Define fully connected layers with the correct size
        self.fc1 = nn.Linear(self.flattened_size, 256)  
        self.fc2 = nn.Linear(256, 4)

    def _get_flattened_size(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 256, 256)  # Simulating input size
            sample_output = self.pool(self.relu(self.conv1(sample_input)))
            sample_output = self.pool(self.relu(self.conv2(sample_output)))
            sample_output = self.pool(self.relu(self.conv3(sample_output)))
            return sample_output.view(1, -1).size(1)  # Flattened size

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
