import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(48, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256,256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256,256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256,64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, n_actions)


        # self.fc1 = nn.Linear(56, 64)
        # # self.bn1 = nn.BatchNorm1d(64)
        # self.fc2 = nn.Linear(64, 256)
        # # self.bn2 = nn.BatchNorm1d(256)
        # self.fc3 = nn.Linear(256,256)
        # # self.bn3 = nn.BatchNorm1d(256)
        # self.fc4 = nn.Linear(256,256)
        # # self.bn4 = nn.BatchNorm1d(256)
        # self.fc5 = nn.Linear(256,256)
        # # self.bn5 = nn.BatchNorm1d(256)
        # self.fc6 = nn.Linear(256,64)
        # # self.bn6 = nn.BatchNorm1d(64)
        # self.fc7 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        # print("DQN forward ",x.shape)
        # print("DQN forward ",x.dim())

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))

        return self.fc7(x)