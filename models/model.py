""" Rohan Sanda 2023 """
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    """This is a single output fully connected neural network that can be customized from the config.py file."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)

        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)

        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class NeuralNetworkMod(nn.Module):
    """This is a multi-output fully connected neural network that predicts both sbp and dbp
        See the poster for model architecture diagrams and more details."""
    def __init__(self):
        super(NeuralNetworkMod, self).__init__()
        
        self.linear1 = nn.Linear(200, 100)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(100, 100)
        self.tanh2 = nn.Tanh()
        
        self.branch1 = nn.Linear(100, 1)
        self.branch2 = nn.Linear(100, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        
        return output1, output2

    
class cnn_1d(nn.Module):
    """ 1D CNN single output neural network."""
    def __init__(self):
        super(cnn_1d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, padding="same"),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, padding="same"),
            nn.BatchNorm1d(20),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(1000, 600),
            nn.Tanh(),
            nn.Linear(600, 1)
        )

    def forward(self, x):
        return self.layers(x)

#For debugging
class PrintLayer(nn.Module):
    """I used this to debug the shapes of the tensors at each layer of the network.
        Also used for checking gradients at different stages of the network."""
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class cnn_1d_mod(nn.Module):
    """This is the multi-output version of the 1D CNN above"""
    def __init__(self):
        super(cnn_1d_mod, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, padding="same"),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, padding="same"),
            nn.BatchNorm1d(20),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(1000, 600),
            nn.Tanh()
        )
        self.systolic_linear = nn.Linear(600, 1)  # Linear layer for systolic blood pressure prediction
        self.diastolic_linear = nn.Linear(600, 1)  # Linear layer for diastolic blood pressure prediction

    def forward(self, x):
        print(x.shape)
        shared_features = self.shared_layers(x)
        systolic_prediction = self.systolic_linear(shared_features)
        diastolic_prediction = self.diastolic_linear(shared_features)
        exit(1)
        return systolic_prediction, diastolic_prediction
        

class ResidualBlock(nn.Module):
    """Defines a residual block that can handle different input and output channels.
        The block just contains some more kernel=3 convolutions and batch normalization layers."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        #PrintLayer(),
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Handle input and output channel dimension mismatch 
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        identity = self.downsample(x)
        #residual connection here
        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """This is the multi-output ResNet model that incorporates the Residual Blocks defined above.
        Note, if you change dimensions like the input and output channels, use the PrintLayer class
        that I wrote above to figure out what the new layer dimensions should be."""
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.Sequential(
            ResidualBlock(10, 20),
            ResidualBlock(20, 20),
            ResidualBlock(20, 10)
        )
        self.maxpool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=5, kernel_size=10, padding=1)
        self.bn2 = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(465, 300)
        self.fc2_sbp = nn.Linear(300, 1)
        self.fc2_dbp = nn.Linear(300, 1)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.maxpool(self.residual_blocks(out))
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = torch.flatten(out, start_dim=1)
        out = self.relu(self.fc1(out))
        sbp_predictions = self.fc2_sbp(out)
        dbp_predictions = self.fc2_dbp(out)
        return sbp_predictions, dbp_predictions






