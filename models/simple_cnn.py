from torch import nn
from torch.nn import functional as F
import torch

# simple CNN architecture adapted from https://jimut123.github.io/
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv11 = nn.Conv2d(1, 16, 3, 1)
        self.conv12 = nn.Conv2d(1, 16, 5, 1)
        self.conv13 = nn.Conv2d(1, 16, 7, 1)
        self.conv14 = nn.Conv2d(1, 16, 9, 1)

        self.conv21 = nn.Conv2d(16, 32, 3, 1)
        self.conv22 = nn.Conv2d(16, 32, 5, 1)
        self.conv23 = nn.Conv2d(16, 32, 7, 1)
        self.conv24 = nn.Conv2d(16, 32, 9, 1)

        self.conv31 = nn.Conv2d(32, 64, 3, 1)
        self.conv32 = nn.Conv2d(32, 64, 5, 1)
        self.conv33 = nn.Conv2d(32, 64, 7, 1)
        self.conv34 = nn.Conv2d(32, 64, 9, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc11 = nn.Linear(64*11*11, 256)
        self.fc12 = nn.Linear(64*8*8, 256)
        self.fc13 = nn.Linear(64*5*5, 256)
        self.fc14 = nn.Linear(64*2*2, 256)

        self.fc21 = nn.Linear(256, 128)
        self.fc22 = nn.Linear(256, 128)
        self.fc23 = nn.Linear(256, 128)
        self.fc24 = nn.Linear(256, 128)

        self.fc33 = nn.Linear(128*4,num_classes)


    def forward(self, inp):

        x = F.relu(self.conv11(inp))
        x = F.relu(self.conv21(x))
        x = F.relu(self.maxpool(self.conv31(x)))

        x = x.view(-1,64*11*11)
        x = self.dropout1(x)
        x = F.relu(self.fc11(x))
        x = self.dropout2(x)
        x = self.fc21(x)

        y = F.relu(self.conv12(inp))
        y = F.relu(self.conv22(y))
        y = F.relu(self.maxpool(self.conv32(y)))
        
        y = y.view(-1,64*8*8)
        y = self.dropout1(y)
        y = F.relu(self.fc12(y))
        y = self.dropout2(y)
        y = self.fc22(y)

        z = F.relu(self.conv13(inp))
        z = F.relu(self.conv23(z))
        z = F.relu(self.maxpool(self.conv33(z)))
        
        z = z.view(-1,64*5*5)
        z = self.dropout1(z)
        z = F.relu(self.fc13(z))
        z = self.dropout2(z)
        z = self.fc23(z)

        ze = F.relu(self.conv14(inp))
        ze = F.relu(self.conv24(ze))
        ze = F.relu(self.maxpool(self.conv34(ze)))
        
        ze = ze.view(-1,64*2*2)
        ze = self.dropout1(ze)
        ze = F.relu(self.fc14(ze))
        ze = self.dropout2(ze)
        ze = self.fc24(ze)

        out_f = torch.cat((x, y, z, ze), dim=1)
        
        out = self.fc33(out_f)

        # output = F.log_softmax(out, dim=1)
        return out