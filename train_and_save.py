import torch
from model import CloudNet  # Import your corrected CloudNet model

# Initialize the model
model = CloudNet()

# Save the model state dictionary
torch.save(model.state_dict(), "CloudNet_LiDAR.pth")

print("✅ Model saved successfully as CloudNet_LiDAR.pth")
