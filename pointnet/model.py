import torch.nn as nn
import torch
import torch.nn.functional as F

class TNet(nn.Module):

    def __init__(self,k = 3):
        super(TNet,self).__init__()
        self.k = k
        self.conv_1 = nn.Conv1d(k,64,1)
        self.conv_2 = nn.Conv1d(64,128,1)
        self.conv_3 = nn.Conv1d(128,1024,1)
        self.fc_1 = nn.Linear(1024,512)
        self.fc_2 = nn.Linear(512,256)
        self.fc_3 = nn.Linear(256,k*k)
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

    def forward(self,x):
        bs = x.size()[0]
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = nn.MaxPool1d(x.size()[-1])(x)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)
        x = x.view(-1,self.k,self.k)
        eye = torch.eye(self.k,requires_grad=True).repeat(bs,1,1)
        if torch.cuda.is_available():
            eye = eye.cuda()
        assert(eye.size()==x.size())
        return x + eye


class TransformNet(nn.Module):

    def __init__(self):
        super(TransformNet,self).__init__()
        self.T1 = TNet(3) #input transform
        self.T2 = TNet(64) #feature transform
        self.conv_1 = nn.Conv1d(3,64,1)
        self.conv_2 = nn.Conv1d(64,128,1)
        self.conv_3 = nn.Conv1d(128,1024,1)
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)

    def forward(self,x):
        m3 = self.T1(x)
        # x is of shape (bs,3,N)
        x = x.transpose(1,2)
        # x is now (bs,N,3)
        # apply input transform
        x = torch.bmm(x,m3).transpose(1,2)
        x = F.relu(self.bn_1(self.conv_1(x)))
        # x is now (bs,N,64)
        # similarly for the feature space
        m64 = self.T2(x)
        x = x.transpose(1,2)
        x = torch.bmm(x,m64).transpose(1,2)
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.conv_3(x))
        x = nn.MaxPool1d(x.size()[-1])(x)
        out = x.view(x.size()[0],-1)
        return out,m3,m64
