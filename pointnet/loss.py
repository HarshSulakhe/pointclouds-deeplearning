import torch.nn as nn
import torch
import torch.nn.functional as F

class PointNetLoss(nn.Module):

    def __init__(self,alpha = 0.0001    ):
        super(PointNetLoss,self).__init__()
        self.alpha = alpha

    def forward(self,outputs,labels,m3,m64):
        criterion = nn.NLLLoss()
        bs = outputs.size()[0]
        eye3 = torch.eye(3).repeat(bs,1,1)
        eye64 = torch.eye(64).repeat(bs,1,1)
        if torch.cuda.is_available():
            eye3 = eye3.cuda()
            eye64 = eye64.cuda()
        diff64 = eye64 - torch.bmm(m64,m64.transpose(1,2))
        diff3 = eye3 - torch.bmm(m3,m3.transpose(1,2))
        return criterion(outputs,labels) + self.alpha*(torch.norm(diff3)+torch.norm(diff64)) / float(bs)
