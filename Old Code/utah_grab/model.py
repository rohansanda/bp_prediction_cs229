import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
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

class cnn_1d_mod(nn.Module):
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
        shared_features = self.shared_layers(x)
        systolic_prediction = self.systolic_linear(shared_features)
        diastolic_prediction = self.diastolic_linear(shared_features)
        return systolic_prediction, diastolic_prediction


    
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10, padding="same")
        self.bn1 = nn.BatchNorm1d(10)
        self.tanh = nn.Tanh()
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(10)
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(10)
        #self.maxpool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(1000, 600)
        self.fc2_sbp = nn.Linear(600, 1)
        self.fc2_dbp = nn.Linear(600, 1)

    def forward(self, x):
        print(x.shape)
        out = self.tanh(self.bn1(self.conv1(x)))
        print(out.shape)
        out = self.maxpool1(out)
        print(out.shape)
        identity = out  # Save the input for the skip connection
        out = self.tanh(self.bn2(self.conv2(out)))
        out = self.tanh(self.bn3(self.conv3(out)))
        #identity = self.maxpool2(identity)  # Apply max pooling to the skip connection
        print(identity.shape)
        print(out.shape)
        #identity = identity[:, :, :out.size(2)]  # Adjust the size of the skip connection
        #identity = identity.expand(-1, out.size(1), -1)  # Expand the dimensions of the skip connection
        out = out + identity  # Add the input (skip connection)
        out = self.flatten(out)
        out = self.tanh(self.fc1(out))
        sbp_predictions = self.fc2_sbp(out)
        dbp_predictions = self.fc2_dbp(out)
        return sbp_predictions, dbp_predictions











