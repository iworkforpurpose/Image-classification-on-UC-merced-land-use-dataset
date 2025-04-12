# models/lenet.py
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=21):
        super(LeNet5, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # change input channels to 3 for RGB
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # Calculate the flattened dimension:
        # Input image size is 256, after two conv/pooling layers.
        # One can compute the resulting size manually (or use adaptive pooling).
        # Here we use adaptive pooling to output a fixed size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc_layer = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

if __name__ == "__main__":
    # Quick test
    model = LeNet5(num_classes=21)
    print(model)
